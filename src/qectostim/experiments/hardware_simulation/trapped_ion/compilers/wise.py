# src/qectostim/experiments/hardware_simulation/trapped_ion/compilers/wise.py
"""
WISE grid architecture compiler.

WISE uses a 2D grid of ion traps with SAT-based optimal routing
by default, or greedy junction-based routing when configured.
"""
from __future__ import annotations

import logging
import math
from typing import (
    Dict,
    List,
    Optional,
    Tuple,
    Any,
    TYPE_CHECKING,
)

import numpy as np

from qectostim.experiments.hardware_simulation.core.pipeline import (
    NativeCircuit,
    MappedCircuit,
    RoutedCircuit,
    ScheduledCircuit,
    ScheduledOperation,
    QubitMapping,
)
from qectostim.experiments.hardware_simulation.core.gates import (
    GateSpec,
    GateType,
)
from qectostim.experiments.hardware_simulation.core.operations import (
    GateOperation,
    TransportOperation,
    OperationType,
)
from qectostim.experiments.hardware_simulation.trapped_ion.compilers.base import (
    TrappedIonCompiler,
    DecomposedGate,
)
from qectostim.experiments.hardware_simulation.trapped_ion.physics import (
    DEFAULT_CALIBRATION as _CAL,
)

if TYPE_CHECKING:
    from qectostim.experiments.hardware_simulation.trapped_ion.architecture import (
        WISEArchitecture,
    )
    from qectostim.experiments.hardware_simulation.trapped_ion.routing import (
        WISERoutingConfig,
    )


_logger = logging.getLogger(__name__)


class WISECompiler(TrappedIonCompiler):
    """Compiler for WISE grid architecture.
    
    WISE uses a 2D grid of ion traps with SAT-based optimal routing
    by default, or greedy junction-based routing when
    ``use_junction_routing=True``.

    Parameters
    ----------
    architecture : WISEArchitecture
        Target WISE grid architecture.
    optimization_level : int
        Optimisation aggressiveness (0–2).
    use_global_rotations : bool
        Whether to use global rotation gates.
    routing_config : WISERoutingConfig or None
        SAT-routing configuration (ignored when junction routing is on).
    use_junction_routing : bool
        If ``True``, use the greedy junction-based router (ported from
        old ``ionRouting()``) instead of the SAT solver.  This is
        faster and more faithful to the old pipeline.
    partition_strategy : str
        Qubit-to-grid mapping strategy.  ``"gate_affinity"`` (default)
        clusters qubits by 2Q interaction frequency.
        ``"spatial"`` replicates the old pipeline's
        ``regularPartition`` + ``arrangeClusters`` using qubit
        coordinates from the Stim circuit and Hungarian assignment.
    """
    
    def __init__(
        self,
        architecture: "WISEArchitecture",
        optimization_level: int = 1,
        use_global_rotations: bool = True,
        routing_config: Optional["WISERoutingConfig"] = None,
        use_junction_routing: bool = False,
        partition_strategy: str = "gate_affinity",
    ):
        super().__init__(architecture, optimization_level, use_global_rotations)
        self.routing_config = routing_config
        self.use_junction_routing = use_junction_routing
        self.partition_strategy = partition_strategy
        self._routing_pass = None
    
    def _setup_passes(self) -> None:
        """Set up WISE compilation passes."""
        pass
    
    def decompose_to_native(self, circuit) -> NativeCircuit:
        """Decompose circuit to native MS + rotation gates.
        
        Builds a stim_instruction_map that maps each stim instruction
        index (in the flattened circuit) to the native op indices it
        produced.  Annotations (TICK, DETECTOR, etc.) are skipped
        during decomposition but preserved in the original circuit
        via stim_source.
        """
        native_ops = []
        stim_instruction_map: Dict[int, List[int]] = {}
        stim_idx = 0
        
        for instruction in circuit.flattened():
            gate_name = instruction.name
            
            # Skip annotations — they stay in the original circuit
            if gate_name in ("TICK", "QUBIT_COORDS", "DETECTOR",
                             "OBSERVABLE_INCLUDE", "SHIFT_COORDS"):
                stim_idx += 1
                continue
            
            targets = instruction.targets_copy()
            qubits = tuple(t.value for t in targets if t.is_qubit_target)
            if not qubits:
                stim_idx += 1
                continue
            
            start_native_idx = len(native_ops)
            decomposed = self.decompose_stim_gate(gate_name, qubits)
            native_ops.extend(decomposed)
            end_native_idx = len(native_ops)
            
            stim_instruction_map[stim_idx] = list(
                range(start_native_idx, end_native_idx)
            )
            stim_idx += 1
        
        return NativeCircuit(
            operations=native_ops,
            num_qubits=circuit.num_qubits,
            metadata={"source": "stim", "compiler": "WISECompiler"},
            stim_instruction_map=stim_instruction_map,
            stim_source=circuit,
        )
    
    def map_qubits(self, circuit: NativeCircuit) -> MappedCircuit:
        """Map logical qubits to WISE grid positions.

        Dispatch based on ``partition_strategy``:

        * ``"gate_affinity"`` (default): clusters qubits by 2Q
          interaction frequency — fast, good for SAT routing.
        * ``"spatial"``: replicates the old pipeline's
          ``regularPartition`` + ``arrangeClusters`` using qubit
          coordinates from the Stim circuit and the Hungarian
          algorithm for optimal cluster-to-trap assignment.

        The ``default_chain_length`` is set to the architecture's trap
        capacity *k* so that downstream fidelity calculations use the
        correct chain length (including spectator ions).
        """
        if self.partition_strategy == "spatial":
            return self._map_qubits_spatial(circuit)
        return self._map_qubits_gate_affinity(circuit)

    def _map_qubits_gate_affinity(self, circuit: NativeCircuit) -> MappedCircuit:
        """Map using gate-affinity clustering (default strategy)."""
        from qectostim.experiments.hardware_simulation.trapped_ion.architecture import (
            WISEArchitecture,
        )

        num_qubits = circuit.num_qubits
        mapping = QubitMapping()

        # Determine chain length from architecture
        chain_length: Optional[int] = None
        n_rows = n_cols = 1
        k = 2  # default capacity
        if isinstance(self.architecture, WISEArchitecture):
            chain_length = self.architecture.k  # trap capacity
            k = self.architecture.k
            n_rows, n_cols = self.architecture.grid_shape

        # --- Gate-affinity clustering ---
        # Count how many 2Q interactions each pair of qubits has.
        interaction_count: Dict[Tuple[int, int], int] = {}
        for op in circuit.operations:
            if len(op.qubits) == 2:
                pair = (min(op.qubits), max(op.qubits))
                interaction_count[pair] = interaction_count.get(pair, 0) + 1

        # Sort pairs by interaction count (most interacting first)
        sorted_pairs = sorted(
            interaction_count.items(), key=lambda x: x[1], reverse=True
        )

        total_slots = n_rows * n_cols
        # Blocks: groups of k consecutive columns per row
        num_blocks_per_row = max(1, n_cols // k)

        # Track which block (row, block_idx) each qubit is assigned to
        qubit_assignment: Dict[int, int] = {}  # qubit → grid position
        block_occupancy: Dict[Tuple[int, int], List[int]] = {}  # (row, blk) → qubits
        for r in range(n_rows):
            for b in range(num_blocks_per_row):
                block_occupancy[(r, b)] = []

        def _try_assign(qubit: int, row: int, blk: int) -> bool:
            """Try to assign qubit to a specific block."""
            if qubit in qubit_assignment:
                return False
            block = block_occupancy.get((row, blk))
            if block is None or len(block) >= k:
                return False
            col = blk * k + len(block)
            if col >= n_cols:
                return False
            pos = row * n_cols + col
            qubit_assignment[qubit] = pos
            block.append(qubit)
            return True

        # Greedily place interacting pairs into same block
        for (q1, q2), _count in sorted_pairs:
            if q1 in qubit_assignment and q2 in qubit_assignment:
                continue
            if q1 in qubit_assignment:
                # Place q2 near q1
                pos1 = qubit_assignment[q1]
                r1 = pos1 // n_cols
                b1 = (pos1 % n_cols) // k
                if not _try_assign(q2, r1, b1):
                    # Try adjacent blocks in same row
                    for db in [1, -1, 2, -2]:
                        nb = b1 + db
                        if 0 <= nb < num_blocks_per_row:
                            if _try_assign(q2, r1, nb):
                                break
            elif q2 in qubit_assignment:
                pos2 = qubit_assignment[q2]
                r2 = pos2 // n_cols
                b2 = (pos2 % n_cols) // k
                if not _try_assign(q1, r2, b2):
                    for db in [1, -1, 2, -2]:
                        nb = b2 + db
                        if 0 <= nb < num_blocks_per_row:
                            if _try_assign(q1, r2, nb):
                                break
            else:
                # Neither assigned — find first block with 2 free slots
                placed = False
                for r in range(n_rows):
                    for b in range(num_blocks_per_row):
                        occ = block_occupancy.get((r, b), [])
                        if len(occ) <= k - 2:
                            _try_assign(q1, r, b)
                            _try_assign(q2, r, b)
                            placed = True
                            break
                    if placed:
                        break

        # Fill remaining qubits sequentially into free slots
        for q in range(num_qubits):
            if q not in qubit_assignment:
                for r in range(n_rows):
                    for b in range(num_blocks_per_row):
                        if _try_assign(q, r, b):
                            break
                    if q in qubit_assignment:
                        break
                # Last resort: assign to next available grid position
                if q not in qubit_assignment:
                    for pos in range(total_slots):
                        if pos not in qubit_assignment.values():
                            qubit_assignment[q] = pos
                            break

        # Build mapping from assignments, including zone information
        for logical_q in range(num_qubits):
            physical_q = qubit_assignment.get(logical_q, logical_q)
            # Compute (row, col) from grid position for zone assignment
            row = physical_q // n_cols if n_cols > 0 else 0
            col = physical_q % n_cols if n_cols > 0 else physical_q
            zone_id = f"trap_{row}_{col}"
            mapping.assign(logical_q, physical_q, zone_id)

        # --- Spectator ion filling ---
        # The WISE grid must be fully populated for the SAT solver to
        # permute correctly.  Fill every empty slot with a unique
        # spectator index (> max data qubit) so the grid is n×(m*k).
        # This also ensures chain_length reflects the true number of
        # ions in each trap segment (including spectators), which is
        # critical for the MS-gate fidelity formula.
        total_grid_slots = n_rows * n_cols
        used_positions = set(mapping.logical_to_physical.values())
        next_spectator_idx = max(used_positions) + 1 if used_positions else num_qubits
        spectator_indices: List[int] = []
        for pos in range(total_grid_slots):
            if pos not in used_positions:
                row = pos // n_cols if n_cols > 0 else 0
                col = pos % n_cols if n_cols > 0 else pos
                zone_id = f"trap_{row}_{col}"
                # Map spectator as a "virtual" logical qubit → physical position
                mapping.assign(next_spectator_idx, pos, zone_id)
                spectator_indices.append(next_spectator_idx)
                next_spectator_idx += 1

        metadata: Dict[str, Any] = {"mapping_strategy": "gate_affinity_wise"}
        if chain_length is not None:
            metadata["default_chain_length"] = chain_length
        if spectator_indices:
            metadata["spectator_indices"] = spectator_indices
            metadata["num_spectators"] = len(spectator_indices)
            metadata["total_ions"] = num_qubits + len(spectator_indices)

        return MappedCircuit(
            native_circuit=circuit,
            mapping=mapping,
            metadata=metadata,
        )
    
    def route(self, circuit: MappedCircuit) -> RoutedCircuit:
        """Route ions through the WISE grid.

        Two strategies are available:

        * **SAT routing** (default): Uses ``WISERoutingPass`` with a
          SAT-based encoder.  Optimal but can be slow for large grids.
        * **Junction routing** (``use_junction_routing=True``): Uses
          the greedy ``route_ions_junction()`` algorithm ported from
          the old ``ionRouting()`` function.  Faster and produces
          results comparable to the old pipeline.

        After routing, 1Q gates / measurements / resets are converted
        to physical operations so the scheduler sees them.
        """
        if self.use_junction_routing:
            return self._route_junction(circuit)
        return self._route_sat(circuit)

    def _route_sat(self, circuit: MappedCircuit) -> RoutedCircuit:
        """SAT-based WISE routing (original path)."""
        from qectostim.experiments.hardware_simulation.trapped_ion.routing import (
            WISERoutingPass,
            WISERoutingConfig,
        )
        
        config = self.routing_config or WISERoutingConfig()
        routing_pass = WISERoutingPass(
            architecture=self.architecture,
            config=config,
        )
        
        routed = routing_pass.route(circuit, self.architecture)

        # The routing pass produces transport + 2Q (MS) gate ops.
        # It may also emit partial 1Q ops via _one_qubit_ops_per_batch,
        # but those are incomplete (missing measurements/resets and
        # emitted as GateOperation instead of proper types).
        # Strip any 1Q GateOperations that the routing pass emitted,
        # then use _interleave_non_2q_ops to insert ALL non-2Q ops at
        # their correct positions.
        filtered_ops = [
            op for op in routed.operations
            if not (
                isinstance(op, GateOperation)
                and op.operation_type == OperationType.GATE_1Q
            )
        ]
        routed.operations = self._interleave_non_2q_ops(filtered_ops, circuit)

        return routed

    def _route_junction(self, circuit: MappedCircuit) -> RoutedCircuit:
        """Greedy junction-based routing (old ``ionRouting()`` port).

        Uses the WISE physical topology graph (traps + junctions +
        crossings) built by ``WISEArchitecture._build_wise_topology()``.

        Steps:
        1. Extract 2Q gate requests with ancilla/data ion references.
        2. Call ``route_ions_junction()`` which greedily executes
           co-located gates, then routes remaining ions through
           shortest paths.
        3. Convert transport ops to physical operations.
        4. Attach 1Q / measurement ops as well.
        """
        from qectostim.experiments.hardware_simulation.trapped_ion.architecture import (
            WISEArchitecture,
        )
        from qectostim.experiments.hardware_simulation.trapped_ion.routing.greedy import (
            route_ions_junction,
            GateRequest,
        )

        arch = self.architecture
        if not isinstance(arch, WISEArchitecture):
            raise TypeError(
                "_route_junction requires a WISEArchitecture, "
                f"got {type(arch).__name__}"
            )

        graph = arch.qccd_graph

        # --- Build GateRequest objects from the native circuit ---
        gate_requests: List[GateRequest] = []
        gate_id = 0
        for dg in circuit.native_circuit.operations:
            if len(dg.qubits) != 2:
                continue
            q1, q2 = dg.qubits
            # Map logical → physical positions
            p1 = circuit.mapping.logical_to_physical.get(q1, q1)
            p2 = circuit.mapping.logical_to_physical.get(q2, q2)

            ion1 = arch.get_ion(p1)
            ion2 = arch.get_ion(p2)
            if ion1 is None or ion2 is None:
                gate_id += 1
                continue

            # In the old code, ancilla = measurement ion, data = data ion.
            # Convention: sort by label; the one labelled 'D' is data.
            # For now, use ion with the smaller index as "ancilla" (the
            # one that moves) — this matches the old sorting heuristic.
            ancilla, data = (ion1, ion2) if ion1.idx <= ion2.idx else (ion2, ion1)
            gate_requests.append(GateRequest(
                ancilla_ion=ancilla,
                data_ion=data,
                priority=gate_id,
                gate_id=gate_id,
            ))
            gate_id += 1

        if not gate_requests:
            return RoutedCircuit(
                operations=[],
                final_mapping=circuit.mapping.copy(),
                routing_overhead=0,
                mapped_circuit=circuit,
            )

        # --- Run junction routing ---
        result = route_ions_junction(
            graph=graph,
            gate_requests=gate_requests,
            trap_capacity=arch.k,
        )

        # --- Convert to physical operations ---
        # First build transport + MS gate ops only (no 1Q/meas/reset).
        # Then use _interleave_non_2q_ops to insert pre/post rotations,
        # measurements, and resets at their correct native-circuit
        # positions — matching the old code's drain-loop pattern.
        routed_ops: List = []

        # Transport ops — extract source/target from concrete op type
        from qectostim.experiments.hardware_simulation.trapped_ion.transport import (
            Split, Merge, Move, JunctionCrossing, CrystalRotation,
        )
        for top in result.transport_ops:
            # Determine source_zone and target_zone from the concrete type
            if isinstance(top, Split):
                src_z, tgt_z = str(top.trap_idx), str(top.crossing_idx)
            elif isinstance(top, Merge):
                src_z, tgt_z = str(top.crossing_idx), str(top.trap_idx)
            elif isinstance(top, Move):
                src_z, tgt_z = str(top.crossing_idx), str(top.crossing_idx)
            elif isinstance(top, JunctionCrossing):
                src_z, tgt_z = str(top.crossing_idx), str(top.junction_idx)
            elif isinstance(top, CrystalRotation):
                src_z = tgt_z = str(top.trap_idx)
            else:
                src_z = tgt_z = "?"
            routed_ops.append(TransportOperation(
                qubit=getattr(top, 'ion_idx', -1),
                source_zone=src_z,
                target_zone=tgt_z,
                duration=top.time_s * 1e6,  # seconds → μs
            ))

        # Gate ops (in execution order) — only MS gates, no rotations
        ms_spec = GateSpec(
            name="MS", gate_type=GateType.TWO_QUBIT,
            num_qubits=2, is_native=True,
        )
        for gid in result.gate_execution_order:
            req = gate_requests[gid]
            gate_op = GateOperation(
                gate=ms_spec,
                qubits=(req.ancilla_ion.idx, req.data_ion.idx),
                duration=_CAL.ms_gate_time * 1e6,
                base_fidelity=_CAL.gate_fidelities().get("MS", 0.99),
                metadata={"routing": "junction"},
            )
            routed_ops.append(gate_op)

        # Interleave 1Q gates, measurements, and resets at correct
        # positions relative to the MS gates.
        all_operations = self._interleave_non_2q_ops(routed_ops, circuit)

        routing_metadata = {
            "routing_strategy": "junction",
            "barriers": result.barriers,
            "total_transport_time_s": result.total_time_s,
            "total_transport_heating": result.total_heating,
            "num_transport_ops": len(result.transport_ops),
            "num_barriers": len(result.barriers),
        }

        return RoutedCircuit(
            operations=all_operations,
            final_mapping=circuit.mapping.copy(),
            routing_overhead=len(result.transport_ops),
            mapped_circuit=circuit,
            metadata=routing_metadata,
        )
    
    def schedule(self, circuit: RoutedCircuit) -> ScheduledCircuit:
        """Schedule operations with WISE-specific parallelization."""
        from qectostim.experiments.hardware_simulation.trapped_ion.scheduling import (
            WISEBatchScheduler,
            BarrierAwareScheduler,
        )

        # Use barrier-aware scheduling when barriers are present
        # (e.g. from junction routing or stim TICK instructions).
        barriers = (circuit.metadata or {}).get("barriers", [])
        if barriers:
            scheduler = BarrierAwareScheduler()
            batches = scheduler.schedule(
                circuit.operations,
                constraints={"barriers": barriers},
            )
            sched_name = "BarrierAwareScheduler"
        else:
            scheduler = WISEBatchScheduler()
            batches = scheduler.schedule(circuit.operations)
            sched_name = "WISEBatchScheduler"

        scheduled_ops, layers, total_dur = self._batches_to_scheduled(batches)

        # Compute per-qubit idle dephasing from scheduled timeline
        dephasing_info = self._compute_dephasing(scheduled_ops, total_dur)

        return ScheduledCircuit(
            layers=layers,
            scheduled_ops=scheduled_ops,
            routed_circuit=circuit,
            batches=batches,
            total_duration=total_dur,
            metadata={
                "scheduler": sched_name,
                "num_barriers": len(barriers),
                "dephasing": dephasing_info,
            },
        )

    # -----------------------------------------------------------------
    # Spatial ion partitioning (old-pipeline compatible mapping)
    # -----------------------------------------------------------------

    def _map_qubits_spatial(self, circuit: NativeCircuit) -> MappedCircuit:
        """Map using spatial partitioning + Hungarian assignment.

        Replicates the old pipeline's ``regularPartition`` followed by
        ``arrangeClusters`` / ``hillClimbOnArrangeClusters``.

        1. Extract (x, y) coordinates from the Stim circuit's
           ``QUBIT_COORDS`` instructions.
        2. Recursively median-split qubits into clusters of size ≤ *k*
           (old ``_partitionClusterIons``).
        3. If there are more clusters than grid traps, merge nearest
           clusters (old ``_merge_clusters_to_limit``).
        4. Use the Hungarian algorithm to optimally assign cluster
           centroids to physical grid trap positions (old
           ``_arrangeClusters``).
        5. Build the ``QubitMapping`` from the result.
        """
        from scipy.optimize import linear_sum_assignment
        from scipy.spatial import distance_matrix as scipy_dist_matrix
        from qectostim.experiments.hardware_simulation.trapped_ion.architecture import (
            WISEArchitecture,
        )

        num_qubits = circuit.num_qubits
        mapping = QubitMapping()

        # --- Architecture parameters ---
        chain_length: Optional[int] = None
        n_rows = n_cols = 1
        k = 2
        if isinstance(self.architecture, WISEArchitecture):
            chain_length = self.architecture.k
            k = self.architecture.k
            n_rows, n_cols = self.architecture.grid_shape

        # --- 1. Extract qubit coordinates from Stim source ---
        qubit_coords: Dict[int, Tuple[float, float]] = {}
        if circuit.stim_source is not None:
            for instr in circuit.stim_source.flattened():
                if instr.name == "QUBIT_COORDS":
                    args = instr.gate_args_copy()
                    targets = instr.targets_copy()
                    if len(args) >= 2 and targets:
                        for t in targets:
                            q = t.value
                            qubit_coords[q] = (args[0], args[1])

        # Fallback: if no coords found, use sequential placement
        if len(qubit_coords) < num_qubits:
            for q in range(num_qubits):
                if q not in qubit_coords:
                    qubit_coords[q] = (float(q % n_cols), float(q // n_cols))

        # --- 2. Two-phase clustering (port of old regularPartition) ---
        # The old code distinguishes measurement (ancilla) vs data qubits:
        # data qubits are clustered into groups of ≤ dIonsPerTrap, then
        # each measurement qubit is attached to the nearest data cluster.
        # This produces more physically realistic trap assignments.
        coords_array = np.array(
            [list(qubit_coords.get(q, (0.0, 0.0))) for q in range(num_qubits)]
        )

        def _partition_recursive(
            indices: List[int],
            coords: np.ndarray,
            capacity: int,
        ) -> List[Tuple[List[int], np.ndarray]]:
            """Recursive median-split partitioning (old _partitionClusterIons)."""
            partitions: List[List[int]] = [list(indices)]
            split_x = True

            while max(len(p) for p in partitions) > capacity:
                to_split = [p for p in partitions if len(p) > capacity]
                for p in to_split:
                    axis = 0 if split_x else 1
                    vals = [float(coords[i][axis]) for i in p]
                    median = float(np.mean(vals))
                    lo, hi = [], []
                    for i, v in zip(p, vals):
                        if v <= median:
                            lo.append(i)
                        else:
                            hi.append(i)
                    if lo:
                        partitions.append(lo)
                    if hi:
                        partitions.append(hi)
                    partitions.remove(p)
                split_x = not split_x

            clusters = []
            for p in partitions:
                c = np.mean(coords[p], axis=0) if p else np.array([0.0, 0.0])
                clusters.append((p, c))
            return clusters

        # Phase 1: Identify measurement vs data qubits from Stim source.
        measured_qubits: set = set()
        if circuit.stim_source is not None:
            for instr in circuit.stim_source.flattened():
                if instr.name in ("M", "MR", "MX", "MY", "MZ", "MRX", "MRY"):
                    for t in instr.targets_copy():
                        if hasattr(t, "value"):
                            measured_qubits.add(t.value)

        data_qubit_indices = sorted(set(range(num_qubits)) - measured_qubits)
        meas_qubit_indices = sorted(measured_qubits & set(range(num_qubits)))

        # Phase 2: regularPartition — cluster data qubits first, then
        # attach each measurement qubit to the nearest data cluster.
        # For WISE arch, effective capacity == k (no -1 adjustment).
        eff_capacity = k

        if data_qubit_indices and meas_qubit_indices:
            d_ions_per_trap = k
            while True:
                # Cluster data qubits into groups of ≤ d_ions_per_trap
                clusters_d = _partition_recursive(
                    data_qubit_indices, coords_array, d_ions_per_trap
                )
                # Cluster measurement qubits individually (1 per cluster)
                clusters_m = _partition_recursive(
                    meas_qubit_indices, coords_array, 1
                )
                # Attach each measurement cluster to the nearest data cluster
                clusters = list(clusters_d)
                for c_m in clusters_m:
                    if not clusters:
                        clusters.append(c_m)
                        continue
                    nearest_idx = min(
                        range(len(clusters)),
                        key=lambda ci: float(
                            np.sum((clusters[ci][1] - c_m[1]) ** 2)
                        ),
                    )
                    merged_ions = clusters[nearest_idx][0] + c_m[0]
                    r_d = len(clusters[nearest_idx][0]) / len(merged_ions)
                    new_centre = c_m[1] * (1 - r_d) + clusters[nearest_idx][1] * r_d
                    clusters[nearest_idx] = (merged_ions, new_centre)

                max_cluster_size = (
                    max(len(c[0]) for c in clusters) if clusters else 0
                )
                if max_cluster_size > eff_capacity:
                    if d_ions_per_trap <= 2:
                        # Fallback: cluster all qubits uniformly
                        all_qubits = list(range(num_qubits))
                        clusters = _partition_recursive(
                            all_qubits, coords_array, eff_capacity
                        )
                        break
                    d_ions_per_trap -= 1
                else:
                    break
        else:
            # No distinction possible — cluster all uniformly
            clusters = _partition_recursive(
                list(range(num_qubits)), coords_array, k
            )

        # --- 3. Merge excess clusters ---
        num_traps = n_rows * (n_cols // k) if k > 0 else n_rows * n_cols
        if len(clusters) > num_traps and num_traps > 0:
            clusters = self._merge_clusters_to_limit(
                clusters, num_traps, k, coords_array
            )

        # --- 4. Hungarian assignment of clusters → grid positions ---
        # Grid trap positions: each trap spans k consecutive columns.
        num_blocks_per_row = max(1, n_cols // k)
        trap_positions: List[Tuple[float, float]] = []
        for r in range(n_rows):
            for b in range(num_blocks_per_row):
                # Use block centroid as the trap position
                col_start = b * k
                col_mid = col_start + (k - 1) / 2.0
                trap_positions.append((col_mid, float(r)))

        if clusters and trap_positions:
            # Pad clusters list with empties if fewer than traps
            while len(clusters) < len(trap_positions):
                clusters.append(([], np.array([0.0, 0.0])))

            # Use hill-climb biasY sweep (faithful port of old code)
            best_map = self._hill_climb_on_arrange_clusters(
                clusters, trap_positions, nearest_neighbour_count=4,
            )

            # Build position → trap-index lookup
            pos_to_trap: Dict[Tuple[float, float], int] = {
                (round(tp[0], 6), round(tp[1], 6)): ti
                for ti, tp in enumerate(trap_positions)
            }

            # Build qubit assignments from cluster → trap mapping
            qubit_assignment: Dict[int, int] = {}
            for ci, mapped_pos in enumerate(best_map):
                if ci >= len(clusters):
                    continue
                qubit_indices = clusters[ci][0]
                key = (round(mapped_pos[0], 6), round(mapped_pos[1], 6))
                ti = pos_to_trap.get(key)
                if ti is None:
                    # Fallback: find nearest trap position
                    ti = min(
                        range(len(trap_positions)),
                        key=lambda t: (trap_positions[t][0] - mapped_pos[0]) ** 2
                        + (trap_positions[t][1] - mapped_pos[1]) ** 2,
                    )
                trap_row = ti // num_blocks_per_row
                trap_blk = ti % num_blocks_per_row
                for slot, q in enumerate(qubit_indices):
                    col = trap_blk * k + slot
                    pos = trap_row * n_cols + col
                    qubit_assignment[q] = pos
        else:
            qubit_assignment = {q: q for q in range(num_qubits)}

        # --- 5. Build mapping ---
        for logical_q in range(num_qubits):
            physical_q = qubit_assignment.get(logical_q, logical_q)
            row = physical_q // n_cols if n_cols > 0 else 0
            col = physical_q % n_cols if n_cols > 0 else physical_q
            zone_id = f"trap_{row}_{col}"
            mapping.assign(logical_q, physical_q, zone_id)

        # --- Spectator filling ---
        total_grid_slots = n_rows * n_cols
        used_positions = set(mapping.logical_to_physical.values())
        next_spectator_idx = max(used_positions) + 1 if used_positions else num_qubits
        spectator_indices: List[int] = []
        for pos in range(total_grid_slots):
            if pos not in used_positions:
                row = pos // n_cols if n_cols > 0 else 0
                col = pos % n_cols if n_cols > 0 else pos
                zone_id = f"trap_{row}_{col}"
                mapping.assign(next_spectator_idx, pos, zone_id)
                spectator_indices.append(next_spectator_idx)
                next_spectator_idx += 1

        metadata: Dict[str, Any] = {"mapping_strategy": "spatial_partition"}
        if chain_length is not None:
            metadata["default_chain_length"] = chain_length
        if spectator_indices:
            metadata["spectator_indices"] = spectator_indices
            metadata["num_spectators"] = len(spectator_indices)
            metadata["total_ions"] = num_qubits + len(spectator_indices)

        return MappedCircuit(
            native_circuit=circuit,
            mapping=mapping,
            metadata=metadata,
        )

    @staticmethod
    def _merge_clusters_to_limit(
        clusters: List[Tuple[List[int], "np.ndarray"]],
        max_clusters: int,
        capacity: int,
        coords: "np.ndarray",
    ) -> List[Tuple[List[int], "np.ndarray"]]:
        """Merge nearest clusters until count ≤ max_clusters.

        Port of old ``_merge_clusters_to_limit``.
        """
        while len(clusters) > max_clusters:
            best_pair = None
            best_dist = float("inf")
            for i in range(len(clusters)):
                for j in range(i + 1, len(clusters)):
                    if len(clusters[i][0]) + len(clusters[j][0]) > capacity:
                        continue
                    d = float(np.sum((clusters[i][1] - clusters[j][1]) ** 2))
                    if d < best_dist:
                        best_pair = (i, j)
                        best_dist = d

            if best_pair is None:
                # Try dissolution: remove smallest cluster, redistribute
                donor_idx = min(
                    range(len(clusters)), key=lambda x: len(clusters[x][0])
                )
                donor_ions, donor_centre = clusters[donor_idx]
                if not donor_ions:
                    clusters.pop(donor_idx)
                    continue
                receivers = [
                    (j, capacity - len(clusters[j][0]), clusters[j][1])
                    for j in range(len(clusters))
                    if j != donor_idx and len(clusters[j][0]) < capacity
                ]
                if not receivers or sum(r[1] for r in receivers) < len(donor_ions):
                    break  # Cannot reduce further
                receivers.sort(
                    key=lambda t: float(np.sum((donor_centre - t[2]) ** 2))
                )
                remaining = list(donor_ions)
                for j, free, _ in receivers:
                    if not remaining:
                        break
                    take = min(free, len(remaining))
                    moved = remaining[:take]
                    remaining = remaining[take:]
                    old_ions, old_c = clusters[j]
                    new_ions = old_ions + moved
                    new_c = np.mean(coords[new_ions], axis=0) if new_ions else old_c
                    clusters[j] = (new_ions, new_c)
                clusters.pop(donor_idx)
                continue

            i, j = best_pair
            ions_i, c_i = clusters[i]
            ions_j, c_j = clusters[j]
            merged = ions_i + ions_j
            merged_c = np.mean(coords[merged], axis=0) if merged else c_i
            for idx in sorted((i, j), reverse=True):
                clusters.pop(idx)
            clusters.append((merged, merged_c))

        return clusters

    # -----------------------------------------------------------------
    # Hill-climb cluster arrangement  (port of old _arrangeClusters /
    # hillClimbOnArrangeClusters from qccd_qubits_to_ions.py)
    # -----------------------------------------------------------------

    _HILL_MAX_ITER = 20_000

    @staticmethod
    def _min_weight_perfect_match(
        A_norm: "np.ndarray",
        B_subset: "np.ndarray",
        centralizer: "np.ndarray",
        divider: "np.ndarray",
        nearest_coords_A: "np.ndarray",
        nearest_dists_A: "np.ndarray",
    ) -> Tuple[float, "np.ndarray"]:
        """Variance-enhanced Hungarian matching.

        Faithful port of old ``_minWeightPerfectMatch``.  The cost matrix
        is Euclidean distance *plus* a variance term that penalises
        assignments that distort the neighbour-distance distribution.
        """
        from scipy.spatial import distance_matrix
        from scipy.optimize import linear_sum_assignment

        rel_B = np.divide((B_subset - centralizer), divider)
        try:
            diffs = (
                np.linalg.norm(
                    nearest_coords_A[:, :, None, :] - rel_B[None, None, :, :],
                    axis=3,
                )
                - nearest_dists_A[:, :, None]
            )
            variance_matrix = np.mean(diffs ** 2, axis=1)
            cost_matrix = distance_matrix(A_norm, rel_B) + variance_matrix
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
        except ValueError:
            cost_matrix = distance_matrix(A_norm, rel_B)
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
        total_cost = float(cost_matrix[row_ind, col_ind].sum())
        return total_cost, col_ind

    @staticmethod
    def _arrange_clusters_biased(
        clusters: List[Tuple[List[int], "np.ndarray"]],
        all_grid_pos: List[Tuple[float, float]],
        nearest_neighbour_count: int = 4,
        bias_y: int = 1,
    ) -> Tuple[List[Tuple[float, float]], float]:
        """Assign clusters to grid positions with biasY weighting.

        Faithful port of old ``_arrangeClusters``.  Returns
        ``(best_map, best_cost)`` where *best_map* is an ordered list of
        grid positions – one per cluster.
        """
        from scipy.spatial import distance_matrix
        from scipy.optimize import linear_sum_assignment

        _MAX_ITER = WISECompiler._HILL_MAX_ITER

        A = np.array([c[1] for c in clusters])
        if len(A) == 0 or len(all_grid_pos) == 0:
            return [], float("inf")

        min_x, min_y = float(A[:, 0].min()), float(A[:, 1].min())
        max_x, max_y = float(A[:, 0].max()), float(A[:, 1].max())
        d_x = max_x - min_x if max_x != min_x else 1.0
        d_y = max_y - min_y if max_y != min_y else 1.0
        centralizer = np.array([[min_x, min_y]] * len(A))
        divider = np.array([[d_x, d_y]] * len(A))
        A_norm = np.divide((A - centralizer), divider)

        # Nearest-neighbour matrices for variance cost
        dist_A = distance_matrix(A_norm, A_norm)
        np.fill_diagonal(dist_A, np.inf)
        nn_count = min(nearest_neighbour_count, len(A) - 1) if len(A) > 1 else 0
        if nn_count > 0:
            nn_indices = np.argsort(dist_A, axis=1)[:, :nn_count]
            nearest_coords = A_norm[nn_indices]
            nearest_dists = dist_A[np.arange(len(A))[:, None], nn_indices]
        else:
            nearest_coords = np.zeros((len(A), 1, 2))
            nearest_dists = np.zeros((len(A), 1))

        grid_arr = np.array(all_grid_pos)
        centroid_B = np.mean(grid_arr, axis=0)
        sorted_to_centroid = sorted(
            all_grid_pos,
            key=lambda p: (p[0] - centroid_B[0]) ** 2 + (p[1] - centroid_B[1]) ** 2,
        )

        # Build anchor candidates around centroid (8 directions + center)
        around_centroid: List[Tuple[float, float]] = []
        for x_sign, y_sign in [
            (-1, 0), (1, 0), (0, 1), (0, -1),
            (0, 0), (1, 1), (-1, -1), (1, -1), (-1, 1),
        ]:
            for p in sorted_to_centroid:
                px, py = p
                cx, cy = float(centroid_B[0]), float(centroid_B[1])
                x_ok = (
                    (x_sign == -1 and px < cx)
                    or (x_sign == 1 and px > cx)
                    or x_sign == 0
                )
                y_ok = (
                    (y_sign == -1 and py < cy)
                    or (y_sign == 1 and py > cy)
                    or y_sign == 0
                )
                if x_ok and y_ok:
                    around_centroid.append(p)
                    break

        around_centroid_set = set(map(tuple, around_centroid))

        best_cost = float("inf")
        best_map: List[Tuple[float, float]] = []

        cardinality_A = len(A_norm)
        if cardinality_A > len(all_grid_pos):
            # Not enough traps — plain Hungarian fallback
            rel_grid = np.divide(
                (grid_arr - centralizer[: len(grid_arr)]),
                divider[: len(grid_arr)],
            )
            cost_simple = distance_matrix(A_norm, rel_grid)
            row_ind, col_ind = linear_sum_assignment(cost_simple)
            return (
                [all_grid_pos[c] for c in col_ind],
                float(cost_simple[row_ind, col_ind].sum()),
            )

        for center in around_centroid_set:
            # Distance-sorted layers with biasY weighting
            not_picked: Dict[float, List[Tuple[float, float]]] = {}
            for p in all_grid_pos:
                dis = max(
                    ((p[0] - center[0]) ** 2) * bias_y,
                    ((p[1] - center[1]) ** 2),
                )
                not_picked.setdefault(dis, []).append(p)

            guaranteed: List[Tuple[float, float]] = []
            next_window: List[Tuple[float, float]] = []
            while not_picked:
                next_window = not_picked.pop(min(not_picked.keys()))
                if len(guaranteed) + len(next_window) < cardinality_A:
                    guaranteed.extend(next_window)
                else:
                    break

            if not next_window:
                continue

            min_wx = min(p[0] for p in next_window)
            min_wy = min(p[1] for p in next_window)
            max_wx = max(p[0] for p in next_window)
            max_wy = max(p[1] for p in next_window)
            d_wx = max_wx - min_wx if max_wx != min_wx else 1.0
            d_wy = max_wy - min_wy if max_wy != min_wy else 1.0
            cent_mat = np.array([[min_wx, min_wy]] * cardinality_A)
            div_mat = np.array([[d_wx, d_wy]] * cardinality_A)

            # Rectangle boundary traversal (old code verbatim logic)
            bottom = sorted(
                [p for p in next_window if p[1] == min_wy], key=lambda p: p[0]
            )
            right = sorted(
                [p for p in next_window if p[0] == max_wx], key=lambda p: p[1]
            )
            top = sorted(
                [p for p in next_window if p[1] == max_wy],
                key=lambda p: p[0],
                reverse=True,
            )
            left = sorted(
                [p for p in next_window if p[0] == min_wx],
                key=lambda p: p[1],
                reverse=True,
            )

            sorted_nw = bottom[:-1] + right[:-1] + top[:-1] + left[:-1]
            if not sorted_nw:
                sorted_nw = list(bottom)

            need = cardinality_A - len(guaranteed)
            if need <= 0:
                continue
            reg_spacing = int(len(sorted_nw) / need)
            if reg_spacing == 0:
                continue

            in_B = [sorted_nw[i * reg_spacing] for i in range(need)]
            card_in_B = len(in_B)

            B_subset = np.array(in_B + guaranteed)
            not_in_B = [w for w in sorted_nw if w not in in_B]

            cost, _ = WISECompiler._min_weight_perfect_match(
                A_norm, B_subset, cent_mat, div_mat, nearest_coords, nearest_dists,
            )
            current_score = cost

            # Hill-climb: swap one point in/out to reduce cost
            _i = 0
            while True:
                next_in_B: list = []
                next_score = float("inf")
                for i in range(card_in_B):
                    for b_not in not_in_B:
                        trial = np.array(
                            in_B[:i] + [b_not] + in_B[i + 1 :] + guaranteed
                        )
                        tc, _ = WISECompiler._min_weight_perfect_match(
                            A_norm, trial, cent_mat, div_mat,
                            nearest_coords, nearest_dists,
                        )
                        if tc < next_score:
                            next_in_B = in_B[:i] + [b_not] + in_B[i + 1 :]
                            next_score = tc
                        if tc == 0:
                            break

                if current_score <= next_score:
                    break
                if _i > _MAX_ITER:
                    break

                in_B = next_in_B
                not_in_B = [w for w in sorted_nw if w not in in_B]
                current_score = next_score
                if current_score == 0:
                    break
                _i += 1

            if current_score < best_cost:
                B_subset = np.array(in_B + guaranteed)
                _, col_ind = WISECompiler._min_weight_perfect_match(
                    A_norm, B_subset, cent_mat, div_mat,
                    nearest_coords, nearest_dists,
                )
                best_map = [tuple(B_subset[idx]) for idx in col_ind]
                best_cost = current_score
            if current_score == 0:
                break

        return best_map, best_cost

    @staticmethod
    def _hill_climb_on_arrange_clusters(
        clusters: List[Tuple[List[int], "np.ndarray"]],
        all_grid_pos: List[Tuple[float, float]],
        nearest_neighbour_count: int = 4,
    ) -> List[Tuple[float, float]]:
        """Sweep biasY and pick best arrangement.

        Faithful port of old ``hillClimbOnArrangeClusters``.  Iterates
        biasY in steps of 10, calling ``_arrange_clusters_biased`` with
        offsets 0..9 at each step, and returns the best result found.
        """
        _MAX_ITER = WISECompiler._HILL_MAX_ITER
        bias_y = 1
        current_cost = float("inf")
        current_map: List[Tuple[float, float]] = []
        next_cost = float("inf")
        next_map: List[Tuple[float, float]] = []

        iters_ = 0
        while True:
            current_map, current_cost = next_map, next_cost
            next_cost = float("inf")
            next_map = []
            best_i = 0
            for i in range(10):
                _map, _cost = WISECompiler._arrange_clusters_biased(
                    clusters, all_grid_pos, nearest_neighbour_count, bias_y + i,
                )
                if _cost < next_cost:
                    best_i = i
                    next_cost = _cost
                    next_map = _map
                if next_cost == 0:
                    break
            if best_i < 10 and next_cost < current_cost:
                return next_map
            if next_cost >= current_cost:
                return current_map if current_map else next_map
            if iters_ > _MAX_ITER:
                return current_map if current_map else next_map
            bias_y += 10
            iters_ += 1

        return current_map  # unreachable but satisfies type checker

    # -----------------------------------------------------------------
    # Dephasing helper
    # -----------------------------------------------------------------

    @staticmethod
    def _compute_dephasing(
        scheduled_ops: List["ScheduledOperation"],
        total_duration: float,
    ) -> Dict[str, Any]:
        """Compute per-qubit idle-time dephasing from a schedule.

        For each qubit, accumulate intervals where the qubit is idle
        (between the end of one operation and the start of the next).
        The dephasing fidelity penalty is ``exp(-idle_time / T2)``.

        This is the new-pipeline equivalent of the old
        ``calculateDephasingFromIdling`` function in scheduling.py.
        """
        T2_S = _CAL.t2_time  # seconds — from CalibrationConstants

        # Build per-qubit timeline of (start, end) intervals
        qubit_intervals: Dict[int, List[Tuple[float, float]]] = {}
        for sop in scheduled_ops:
            op = sop.operation
            qubits: List[int] = []
            if hasattr(op, "qubits") and op.qubits:
                qubits = list(op.qubits) if hasattr(op.qubits, "__iter__") else [op.qubits]
            elif hasattr(op, "qubit") and op.qubit is not None:
                qubits = [op.qubit]
            for q in qubits:
                qubit_intervals.setdefault(q, []).append(
                    (sop.start_time, sop.end_time)
                )

        # Sort intervals and compute idle gaps
        per_qubit_idle_us: Dict[int, float] = {}
        total_idle_us = 0.0
        for q, intervals in qubit_intervals.items():
            intervals.sort()
            idle = 0.0
            last_end = 0.0
            for s, e in intervals:
                if s > last_end:
                    idle += s - last_end
                last_end = max(last_end, e)
            # Idle after last op until schedule end
            if total_duration > last_end:
                idle += total_duration - last_end
            per_qubit_idle_us[q] = idle
            total_idle_us += idle

        # Compute dephasing fidelity per qubit
        per_qubit_fidelity: Dict[int, float] = {}
        for q, idle_us in per_qubit_idle_us.items():
            idle_s = idle_us * 1e-6  # convert μs → s
            per_qubit_fidelity[q] = math.exp(-idle_s / T2_S)

        avg_fidelity = (
            sum(per_qubit_fidelity.values()) / len(per_qubit_fidelity)
            if per_qubit_fidelity
            else 1.0
        )

        return {
            "per_qubit_idle_us": per_qubit_idle_us,
            "per_qubit_fidelity": per_qubit_fidelity,
            "avg_dephasing_fidelity": avg_fidelity,
            "total_idle_us": total_idle_us,
        }
