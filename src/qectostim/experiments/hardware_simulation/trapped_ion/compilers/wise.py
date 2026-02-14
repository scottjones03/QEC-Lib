# src/qectostim/experiments/hardware_simulation/trapped_ion/compilers/wise.py
"""
WISE grid architecture compiler.

WISE uses a 2D grid of ion traps with SAT-based optimal routing
by default, or greedy junction-based routing when configured.
"""
from __future__ import annotations

import concurrent.futures
import logging
import math
import multiprocessing as mp
import os
import time
from concurrent.futures import ProcessPoolExecutor
from typing import (
    Dict,
    List,
    Optional,
    Sequence,
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
        Qubit-to-grid mapping strategy.  ``"classic"`` (default)
        calls the exact old-pipeline ``regularPartition`` +
        ``hillClimbOnArrangeClusters`` functions for bit-exact
        parity with the legacy code.
        ``"gate_affinity"`` clusters qubits by 2Q interaction
        frequency.
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
        partition_strategy: str = "classic",
    ):
        super().__init__(architecture, optimization_level, use_global_rotations)
        self.routing_config = routing_config
        self.use_junction_routing = use_junction_routing
        self.partition_strategy = partition_strategy
        self._routing_pass = None
    
    def _setup_passes(self) -> None:
        """Set up WISE compilation passes."""
        pass
    
    def decompose_to_native(self, circuit, qec_metadata=None) -> NativeCircuit:
        """Decompose circuit to native MS + rotation gates.
        
        Builds a stim_instruction_map that maps each stim instruction
        index (in the flattened circuit) to the native op indices it
        produced.  Annotations (TICK, DETECTOR, etc.) are skipped
        during decomposition but preserved in the original circuit
        via stim_source.

        Parameters
        ----------
        circuit : stim.Circuit
            The Stim circuit to decompose.
        qec_metadata : Optional[QECMetadata]
            Rich QEC metadata to propagate to the NativeCircuit.
            When provided, downstream pipeline stages (map_qubits,
            route) can access stabilizer structure and code geometry.
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
            qec_metadata=qec_metadata,
        )
    
    def map_qubits(self, circuit: NativeCircuit) -> MappedCircuit:
        """Map logical qubits to WISE grid positions.

        Dispatch based on ``partition_strategy``:

        * ``"classic"`` (default): calls the **exact** old-pipeline
          functions ``regularPartition`` + ``hillClimbOnArrangeClusters``
          with Ion objects created from ``QUBIT_COORDS``.  Produces
          identical qubit-to-trap assignments as the legacy code.
        * ``"gate_affinity"``: clusters qubits by 2Q interaction
          frequency — fast, good for SAT routing.
        * ``"spatial"``: re-implementation of the old pipeline's
          ``regularPartition`` using qubit coordinates and Hungarian
          assignment (not bit-exact with old code).
        * ``"stabilizer"``: uses QEC stabilizer structure for
          co-location of stabilizer groups.

        The ``default_chain_length`` is set to the architecture's trap
        capacity *k* so that downstream fidelity calculations use the
        correct chain length (including spectator ions).
        """
        if self.partition_strategy == "classic":
            return self._map_qubits_classic(circuit)
        if self.partition_strategy == "spatial":
            return self._map_qubits_spatial(circuit)
        if self.partition_strategy == "stabilizer":
            return self._map_qubits_stabilizer(circuit)
        if self.partition_strategy == "gate_affinity":
            return self._map_qubits_gate_affinity(circuit)

        # Auto-promote to stabilizer-aware placement when QECMetadata
        # with stabilizer structure is available and the user hasn't
        # explicitly chosen a different strategy.
        meta = getattr(circuit, "qec_metadata", None)
        if meta is not None and getattr(meta, "stabilizers", None):
            _logger.debug("map_qubits: auto-promoting to stabilizer strategy "
                          "(qec_metadata with stabilizer info detected)")
            return self._map_qubits_stabilizer(circuit)

        # Default: classic (matches old pipeline)
        return self._map_qubits_classic(circuit)

    # ------------------------------------------------------------------ #
    #  Classic mapping — exact port of old processCircuitWiseArch logic   #
    # ------------------------------------------------------------------ #

    def _map_qubits_classic(self, circuit: NativeCircuit) -> MappedCircuit:
        """Map using the **exact** old-pipeline algorithm.

        1. Create temporary ``Ion`` objects with positions taken from
           the Stim circuit's ``QUBIT_COORDS`` (same as old
           ``_parseCircuitString``).
        2. Separate measurement vs data ions using coordinate parity
           (even x → measurement, odd x → data), matching the old
           pipeline's heuristic.
        3. Call ``regularPartition(measurementIons, dataIons, k,
           isWISEArch=True, maxClusters=m*n)``.
        4. Call ``hillClimbOnArrangeClusters(clusters, allGridPos)``
           to assign clusters to grid positions.
        5. Build the ``QubitMapping`` by placing each ion into the
           grid slot determined by its cluster's grid assignment.
        """
        from qectostim.experiments.hardware_simulation.trapped_ion.architecture import (
            WISEArchitecture, Ion,
        )
        from qectostim.experiments.hardware_simulation.trapped_ion.routing.qccd_qubits_to_ions import (
            regularPartition, hillClimbOnArrangeClusters,
        )

        arch = self.architecture
        if not isinstance(arch, WISEArchitecture):
            # Fallback for non-WISE architectures
            return self._map_qubits_gate_affinity(circuit)

        num_qubits = circuit.num_qubits
        k = arch.k                        # ions per trap (capacity)
        m = arch.col_groups               # column groups
        n = arch.rows                     # rows
        n_cols = arch.total_columns       # m * k
        chain_length = k

        # --- 1. Extract QUBIT_COORDS from stim source ---
        # Only qubits that have QUBIT_COORDS participate in the
        # mapping (same as old _parseCircuitString — qubits without
        # coords are not ions and don't appear in instructions).
        qubit_coords: Dict[int, Tuple[float, float]] = {}
        stim_src = getattr(circuit, 'stim_source', None)
        if stim_src is not None:
            for instr in stim_src.flattened():
                if instr.name == "QUBIT_COORDS":
                    args = instr.gate_args_copy()
                    targets = instr.targets_copy()
                    if len(args) >= 2 and targets:
                        for t in targets:
                            qubit_coords[t.value] = (args[0], args[1])

        # If no QUBIT_COORDS, fall back to sequential for all qubits
        if not qubit_coords:
            for q in range(num_qubits):
                qubit_coords[q] = (float(q), 0.0)

        # --- 2. Create Ion objects and separate meas / data ---
        #   Old pipeline: (coords[0] % 2) == 0  →  measurement ion
        #                  else                  →  data ion
        measurement_ions: List[Ion] = []
        data_ions: List[Ion] = []
        idx_to_ion: Dict[int, Ion] = {}

        for q_idx in sorted(qubit_coords.keys()):
            cx, cy = qubit_coords[q_idx]
            ion = Ion(idx=q_idx, position=(cx, cy), label="Q")
            idx_to_ion[q_idx] = ion
            if int(cx) % 2 == 0:
                measurement_ions.append(ion)
            else:
                data_ions.append(ion)

        # --- 3. regularPartition ---
        clusters = regularPartition(
            measurement_ions, data_ions, k,
            isWISEArch=True,
            maxClusters=m * n,
        )

        # --- 4. hillClimbOnArrangeClusters ---
        allGridPos: List[Tuple[int, int]] = []
        for r in range(n):
            for c in range(m):
                allGridPos.append((c, r))

        gridPositions = hillClimbOnArrangeClusters(
            clusters, allGridPos=allGridPos,
        )
        # gridPositions[i] = (col, row) for cluster i

        # --- 5. Build qubit → grid-position mapping ---
        # Old pipeline logic:
        #   trap_for_grid[(2*col, row)] = clusters[trapIdx]
        #   ions in that trap get positions row * n_cols + trap_col_offset + slot
        mapping = QubitMapping()
        qubit_assignment: Dict[int, int] = {}

        for trap_idx, (col, row) in enumerate(gridPositions):
            if trap_idx >= len(clusters):
                continue
            cluster_ions = clusters[trap_idx][0]
            for slot, ion in enumerate(cluster_ions):
                # Grid position: row-major with k columns per block
                grid_pos = row * n_cols + col * k + slot
                qubit_assignment[ion.idx] = grid_pos

        # Assign mapped qubits (those with QUBIT_COORDS)
        used_positions: set = set()
        for logical_q in sorted(qubit_assignment.keys()):
            physical_q = qubit_assignment[logical_q]
            row = physical_q // n_cols if n_cols > 0 else 0
            col = physical_q % n_cols if n_cols > 0 else physical_q
            zone_id = f"trap_{row}_{col}"
            mapping.assign(logical_q, physical_q, zone_id)
            used_positions.add(physical_q)

        # Assign unmapped logical qubits (those without QUBIT_COORDS)
        # to remaining free grid positions — they act as spectators.
        total_grid_slots = n * n_cols
        free_positions = sorted(
            pos for pos in range(total_grid_slots)
            if pos not in used_positions
        )
        free_idx = 0
        for logical_q in range(num_qubits):
            if logical_q not in qubit_assignment:
                if free_idx < len(free_positions):
                    physical_q = free_positions[free_idx]
                    free_idx += 1
                else:
                    # Beyond grid capacity — assign an out-of-grid index
                    physical_q = total_grid_slots + logical_q
                row = physical_q // n_cols if n_cols > 0 else 0
                col = physical_q % n_cols if n_cols > 0 else physical_q
                zone_id = f"trap_{row}_{col}"
                mapping.assign(logical_q, physical_q, zone_id)
                used_positions.add(physical_q)

        # --- Spectator filling ---
        # Fill any remaining empty grid slots with virtual spectators
        next_spectator_idx = max(
            max(used_positions) + 1 if used_positions else num_qubits,
            num_qubits,
        )
        spectator_indices: List[int] = []
        for pos in range(total_grid_slots):
            if pos not in used_positions:
                row = pos // n_cols if n_cols > 0 else 0
                col = pos % n_cols if n_cols > 0 else pos
                zone_id = f"trap_{row}_{col}"
                mapping.assign(next_spectator_idx, pos, zone_id)
                spectator_indices.append(next_spectator_idx)
                next_spectator_idx += 1

        metadata: Dict[str, Any] = {"mapping_strategy": "classic"}
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

    def _map_qubits_stabilizer(self, circuit: NativeCircuit) -> MappedCircuit:
        """Map using QEC stabilizer structure — keeps stabilizer
        groups co-located in the same trap block.

        Falls back to gate_affinity when ``qec_metadata`` is absent.
        """
        from qectostim.experiments.hardware_simulation.trapped_ion.architecture import (
            WISEArchitecture,
        )

        meta = getattr(circuit, "qec_metadata", None)
        if meta is None:
            _logger.debug("_map_qubits_stabilizer: no qec_metadata, "
                          "falling back to gate_affinity")
            return self._map_qubits_gate_affinity(circuit)

        num_qubits = circuit.num_qubits
        mapping = QubitMapping()

        chain_length: Optional[int] = None
        n_rows = n_cols = 1
        k = 2
        if isinstance(self.architecture, WISEArchitecture):
            chain_length = self.architecture.k
            k = self.architecture.k
            n_rows, n_cols = self.architecture.grid_shape

        num_blocks_per_row = max(1, n_cols // k)
        block_occupancy: Dict[Tuple[int, int], List[int]] = {}
        for r in range(n_rows):
            for b in range(num_blocks_per_row):
                block_occupancy[(r, b)] = []

        qubit_assignment: Dict[int, int] = {}

        def _try_assign(qubit: int, row: int, blk: int) -> bool:
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

        # Place each stabilizer's support set into the same block
        for stab_info in meta.stabilizers:
            # Flatten supports (List[List[int]]) and combine with ancillas
            flat_supports: set = set()
            for supp in stab_info.supports:
                flat_supports.update(supp)
            all_qubits = sorted(flat_supports | set(stab_info.ancillas))
            # Try to fit the whole group into one block
            placed = False
            for r in range(n_rows):
                for b in range(num_blocks_per_row):
                    occ = block_occupancy.get((r, b), [])
                    # Only count qubits not already assigned
                    new_qubits = [q for q in all_qubits if q not in qubit_assignment]
                    if len(occ) + len(new_qubits) <= k:
                        for q in new_qubits:
                            _try_assign(q, r, b)
                        placed = True
                        break
                if placed:
                    break

        # Fill remaining qubits
        for q in range(num_qubits):
            if q not in qubit_assignment:
                for r in range(n_rows):
                    for b in range(num_blocks_per_row):
                        if _try_assign(q, r, b):
                            break
                    if q in qubit_assignment:
                        break
                if q not in qubit_assignment:
                    total_slots = n_rows * n_cols
                    for pos in range(total_slots):
                        if pos not in qubit_assignment.values():
                            qubit_assignment[q] = pos
                            break

        for logical_q in range(num_qubits):
            physical_q = qubit_assignment.get(logical_q, logical_q)
            row = physical_q // n_cols if n_cols > 0 else 0
            col = physical_q % n_cols if n_cols > 0 else physical_q
            zone_id = f"trap_{row}_{col}"
            mapping.assign(logical_q, physical_q, zone_id)

        # Spectator filling
        total_grid_slots = n_rows * n_cols
        used_positions = set(mapping.logical_to_physical.values())
        next_spec = max(used_positions) + 1 if used_positions else num_qubits
        spectator_indices: List[int] = []
        for pos in range(total_grid_slots):
            if pos not in used_positions:
                row = pos // n_cols if n_cols > 0 else 0
                col = pos % n_cols if n_cols > 0 else pos
                mapping.assign(next_spec, pos, f"trap_{row}_{col}")
                spectator_indices.append(next_spec)
                next_spec += 1

        md: Dict[str, Any] = {"mapping_strategy": "stabilizer_wise"}
        if chain_length is not None:
            md["default_chain_length"] = chain_length
        if spectator_indices:
            md["spectator_indices"] = spectator_indices
            md["num_spectators"] = len(spectator_indices)
            md["total_ions"] = num_qubits + len(spectator_indices)
        return MappedCircuit(native_circuit=circuit, mapping=mapping, metadata=md)

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
        """SAT-based WISE routing — calls ionRoutingWISEArch directly."""
        from qectostim.experiments.hardware_simulation.trapped_ion.routing import (
            ionRoutingWISEArch,
            old_ops_to_transport_list,
        )
        from qectostim.experiments.hardware_simulation.trapped_ion.architecture import (
            WISEArchitecture,
        )
        from qectostim.experiments.hardware_simulation.trapped_ion.operations import (
            MSGate,
            SingleQubitGate,
            Measurement as IonMeasurement,
            QubitReset as IonQubitReset,
        )

        arch = self.architecture
        if not isinstance(arch, WISEArchitecture):
            raise TypeError(
                "_route_sat requires a WISEArchitecture, "
                f"got {type(arch).__name__}"
            )

        # Build the FULL list of qubit operations the router expects.
        # The router's section 4b consumes 1Q ops between MS rounds,
        # and only ops that appear in the input make it into allOps.
        # Without 1Q/M/R the scheduler sees only ReconfigurationPlanners
        # and MS gates, producing an unrealistically low total_duration.
        qi = arch.qubit_ions
        l2p = circuit.mapping.logical_to_physical
        old_operations: List = []
        ms_gates: List[MSGate] = []          # track MS subset for toMoveOps
        for dg in circuit.native_circuit.operations:
            nq = len(dg.qubits)
            if nq == 2:
                q1, q2 = dg.qubits
                p1 = l2p.get(q1, q1)
                p2 = l2p.get(q2, q2)
                if 0 <= p1 < len(qi) and 0 <= p2 < len(qi):
                    gate = MSGate.qubitOperation(qi[p1], qi[p2])
                    old_operations.append(gate)
                    ms_gates.append(gate)
            elif nq == 1:
                q = dg.qubits[0]
                p = l2p.get(q, q)
                if 0 <= p < len(qi):
                    ion = qi[p]
                    if dg.name in ("RX", "RY"):
                        old_operations.append(
                            SingleQubitGate(ion, gate_type=dg.name)
                        )
                    elif dg.name == "M":
                        old_operations.append(IonMeasurement(ion))
                    elif dg.name == "R":
                        old_operations.append(IonQubitReset(ion))
                    else:
                        old_operations.append(
                            SingleQubitGate(ion, gate_type=dg.name)
                        )

        # --- Build toMoveOps from stim circuit CX layers ---
        # The old pipeline's circuitString() grouped CNOT gates by
        # TICK-separated layers in the stim circuit.  Each parallel
        # CX instruction becomes one round of MS gates in toMoveOps.
        # Without toMoveOps the greedy grouper produces incorrect
        # parallelism — the router needs the exact per-round grouping.
        _stim_toMoveOps: Optional[List[List[MSGate]]] = None
        stim_src = getattr(circuit.native_circuit, 'stim_source', None)
        if stim_src is not None:
            _stim_toMoveOps = []
            _current_round: List[MSGate] = []
            _ms_gate_idx = 0  # index into ms_gates (NOT old_operations)
            for instr in stim_src.flattened():
                if instr.name == "TICK":
                    if _current_round:
                        _stim_toMoveOps.append(_current_round)
                        _current_round = []
                elif instr.name == "CX":
                    targets = instr.targets_copy()
                    # Each CX pair is (control, target) → one MS gate
                    for i in range(0, len(targets), 2):
                        if _ms_gate_idx < len(ms_gates):
                            _current_round.append(ms_gates[_ms_gate_idx])
                            _ms_gate_idx += 1
            if _current_round:
                _stim_toMoveOps.append(_current_round)
            # Sanity: if we consumed all MS gates, use this as toMoveOps
            if _ms_gate_idx == len(ms_gates) and _stim_toMoveOps:
                _logger.debug(
                    "_route_sat: built toMoveOps with %d rounds from "
                    "stim CX layers (%d MS gates total)",
                    len(_stim_toMoveOps), _ms_gate_idx,
                )
            else:
                _logger.warning(
                    "_route_sat: stim CX layer extraction mismatch: "
                    "consumed %d / %d MS gates — falling back to None",
                    _ms_gate_idx, len(ms_gates),
                )
                _stim_toMoveOps = None

        # --- QECMetadata-driven hints ---
        toMoveOps = None
        round_repetition_hint = None
        meta = getattr(circuit.native_circuit, "qec_metadata", None)
        if meta is not None:
            # Build toMoveOps from cnot_schedule
            if meta.cnot_schedule:
                _template = self._build_toMoveOps(
                    meta.cnot_schedule,
                    qi,
                    circuit.mapping.logical_to_physical,
                )
                # The template is one stabilizer round's worth of CNOT
                # layers.  Replicate it meta.rounds times so it covers
                # the full circuit (matching the _stim_toMoveOps length).
                _n_repeats = max(1, getattr(meta, 'rounds', 1))
                toMoveOps = _template * _n_repeats
                _logger.debug(
                    "_route_sat: built toMoveOps with %d rounds "
                    "(%d template × %d repeats) from cnot_schedule",
                    len(toMoveOps), len(_template), _n_repeats,
                )
            # Build round_repetition_hint from phases
            if meta.phases:
                round_repetition_hint = self._build_round_repetition_hint(
                    meta.phases,
                )
                _logger.debug(
                    "_route_sat: built round_repetition_hint with %d phases",
                    len(meta.phases),
                )

        # Fallback: when no QECMetadata is available, use the stim
        # CX-layer structure extracted above.
        if toMoveOps is None and _stim_toMoveOps:
            toMoveOps = _stim_toMoveOps

        # Call the WISE SAT router directly with the architecture
        rc = self.routing_config
        subgridsize = getattr(rc, 'subgridsize', (6, 4, 1)) if rc else (6, 4, 1)
        lookahead = getattr(rc, 'lookahead', 2) if rc else 2

        # --- Auto-progress bar when running interactively ----
        progress_cb = getattr(rc, 'progress_callback', None) if rc else None
        _auto_close = None
        if progress_cb is None:
            try:
                # Auto-detect notebook / interactive environment
                get_ipython  # type: ignore[name-defined]
                from qectostim.experiments.hardware_simulation.trapped_ion.routing.config import (
                    make_tqdm_progress_callback,
                )
                progress_cb, _auto_close = make_tqdm_progress_callback("SAT Routing")
            except NameError:
                pass  # not in IPython — no auto-progress

        all_ops, barriers, reconfig_time = ionRoutingWISEArch(
            arch,
            old_operations,
            lookahead=lookahead,
            subgridsize=subgridsize,
            toMoveOps=toMoveOps,
            round_repetition_hint=round_repetition_hint,
            progress_callback=progress_cb,
        )

        # Clean up auto-created progress bar
        if _auto_close is not None:
            _auto_close()

        # Convert old ops to transport list for new pipeline
        ops_list, barrier_list, metadata = old_ops_to_transport_list(
            all_ops, barriers,
        )

        routing_metadata = {
            "routing_strategy": "wise_sat",
            "barriers": barrier_list,
            "reconfig_time": reconfig_time,
            "num_old_ops": len(ops_list),
            "old_operations": ops_list,
            "old_barriers": barrier_list,
        }

        return RoutedCircuit(
            operations=[],  # Gate ops (old ops stored in routing_operations)
            final_mapping=circuit.mapping.copy(),
            routing_overhead=len(ops_list),
            mapped_circuit=circuit,
            routing_operations=ops_list,  # Old ops available via interleaved_operations()
            metadata=routing_metadata,
        )

    @staticmethod
    def _build_toMoveOps(
        cnot_schedule: Dict[str, List[List[Tuple[int, int]]]],
        qubit_ions: Sequence[int],
        logical_to_physical: Dict[int, int],
    ) -> List[List["MSGate"]]:
        """Convert a CNOT schedule dict into ``toMoveOps`` for the SAT
        router.

        Each inner list is one parallel round of MS gates.
        """
        from qectostim.experiments.hardware_simulation.trapped_ion.operations import (
            MSGate,
        )

        rounds: List[List[MSGate]] = []
        for stab_type in ("x", "z"):
            layers = cnot_schedule.get(stab_type)
            if layers is None:
                continue
            for layer in layers:
                ms_round: List[MSGate] = []
                for ctrl, tgt in layer:
                    p1 = logical_to_physical.get(ctrl, ctrl)
                    p2 = logical_to_physical.get(tgt, tgt)
                    if 0 <= p1 < len(qubit_ions) and 0 <= p2 < len(qubit_ions):
                        ms_round.append(
                            MSGate.qubitOperation(qubit_ions[p1], qubit_ions[p2])
                        )
                if ms_round:
                    rounds.append(ms_round)
        return rounds

    @staticmethod
    def _classify_ops_by_block(
        operations: Sequence["MSGate"],
        blocks: Sequence["BlockInfo"],
        qubit_ions: Sequence[int],
        physical_to_logical: Dict[int, int],
    ) -> Dict[str, List["MSGate"]]:
        """Classify MS gates by QEC block name.

        Returns a dict ``{block_name: [MSGate, ...]}`` for each
        block that contains at least one gate.
        """
        # Build a map: logical_qubit -> block_name
        qubit_block: Dict[int, str] = {}
        for blk in blocks:
            for q in blk.data_qubits:
                qubit_block[q] = blk.block_name
            for q in blk.x_ancilla_qubits:
                qubit_block[q] = blk.block_name
            for q in blk.z_ancilla_qubits:
                qubit_block[q] = blk.block_name

        # Reverse ion -> logical
        ion_to_logical: Dict[int, int] = {}
        for phys, logical in physical_to_logical.items():
            if 0 <= phys < len(qubit_ions):
                ion_to_logical[qubit_ions[phys]] = logical

        result: Dict[str, List] = {}
        for gate in operations:
            ions = gate.ions if hasattr(gate, "ions") else []
            block_name = "unknown"
            for ion in ions:
                lq = ion_to_logical.get(ion)
                if lq is not None and lq in qubit_block:
                    block_name = qubit_block[lq]
                    break
            result.setdefault(block_name, []).append(gate)
        return result

    @staticmethod
    def _build_round_repetition_hint(
        phases: Optional[Sequence["PhaseInfo"]],
    ) -> Optional[Dict[str, Any]]:
        """Build a ``round_repetition_hint`` dict from phase info.

        The hint tells the router which phases repeat and which phase
        transitions force a cache-key recheck.

        Returns ``None`` when phase info is unavailable.
        """
        if phases is None or len(phases) == 0:
            return None
        hint_phases: List[Dict[str, Any]] = []
        force_recheck: List[int] = []
        ion_return: bool = False
        # Accumulate MS round count to translate phase boundaries
        # into the MS-round index space the router uses.
        _ms_round_cursor: int = 0
        for i, ph in enumerate(phases):
            # Count how many CNOT layers (MS rounds) this phase
            # contributes.  Each round_signature tuple element is one
            # parallel CNOT layer.
            _sig_layers = len(ph.round_signature) if ph.round_signature else 0
            _phase_ms_rounds = _sig_layers * max(1, ph.num_rounds) if _sig_layers else 0
            entry: Dict[str, Any] = {
                "phase_type": ph.phase_type,
                "num_rounds": ph.num_rounds,
                "is_repeated": ph.is_repeated,
            }
            if ph.identical_to_phase is not None:
                pass  # Cross-phase cache sharing: same pattern as earlier phase
            else:
                # Boundary between distinct phases -> force cache recheck
                # at the MS round index where this phase starts.
                if i > 0 and _ms_round_cursor > 0:
                    force_recheck.append(_ms_round_cursor)
            hint_phases.append(entry)
            if ph.is_repeated:
                ion_return = True
            _ms_round_cursor += _phase_ms_rounds
        return {
            "phases": hint_phases,
            "force_recheck_at": force_recheck,
            "ion_return": ion_return,
        }

    def _route_junction(self, circuit: MappedCircuit) -> RoutedCircuit:
        """Greedy junction-based routing — calls ionRouting() directly.

        Uses the ionRouting() function which greedily executes
        co-located gates, then routes remaining ions through
        shortest paths via the architecture graph.
        """
        from qectostim.experiments.hardware_simulation.trapped_ion.architecture import (
            WISEArchitecture,
        )
        from qectostim.experiments.hardware_simulation.trapped_ion.operations import (
            MSGate,
            SingleQubitGate,
            Measurement as IonMeasurement,
            QubitReset as IonQubitReset,
        )
        from qectostim.experiments.hardware_simulation.trapped_ion.routing import (
            ionRouting,
            old_ops_to_transport_list,
        )

        arch = self.architecture
        if not isinstance(arch, WISEArchitecture):
            raise TypeError(
                "_route_junction requires a WISEArchitecture, "
                f"got {type(arch).__name__}"
            )

        # Pass ALL ops so the junction router sees 1Q gates,
        # measurements, and resets — they appear in allOps and the
        # scheduler computes the correct total_duration.
        qi = arch.qubit_ions
        l2p = circuit.mapping.logical_to_physical
        old_operations = []
        for dg in circuit.native_circuit.operations:
            nq = len(dg.qubits)
            if nq == 2:
                q1, q2 = dg.qubits
                p1 = l2p.get(q1, q1)
                p2 = l2p.get(q2, q2)
                if 0 <= p1 < len(qi) and 0 <= p2 < len(qi):
                    gate = MSGate.qubitOperation(qi[p1], qi[p2])
                    old_operations.append(gate)
            elif nq == 1:
                q = dg.qubits[0]
                p = l2p.get(q, q)
                if 0 <= p < len(qi):
                    ion = qi[p]
                    if dg.name in ("RX", "RY"):
                        old_operations.append(
                            SingleQubitGate(ion, gate_type=dg.name)
                        )
                    elif dg.name == "M":
                        old_operations.append(IonMeasurement(ion))
                    elif dg.name == "R":
                        old_operations.append(IonQubitReset(ion))
                    else:
                        old_operations.append(
                            SingleQubitGate(ion, gate_type=dg.name)
                        )

        # Call the greedy junction router directly with the architecture
        all_ops, barriers = ionRouting(
            arch,
            old_operations,
            trapCapacity=arch.k,
        )

        # Convert old ops to transport list for new pipeline
        ops_list, barrier_list, metadata = old_ops_to_transport_list(
            all_ops, barriers,
        )

        routing_metadata = {
            "routing_strategy": "junction",
            "barriers": barrier_list,
            "num_old_ops": len(ops_list),
            "old_operations": ops_list,
            "old_barriers": barrier_list,
        }

        return RoutedCircuit(
            operations=[],  # Gate ops (old ops stored in routing_operations)
            final_mapping=circuit.mapping.copy(),
            routing_overhead=len(ops_list),
            mapped_circuit=circuit,
            routing_operations=ops_list,  # Old ops available via interleaved_operations()
            metadata=routing_metadata,
        )
    
    def schedule(self, circuit: RoutedCircuit) -> ScheduledCircuit:
        """Schedule operations with WISE-specific parallelization.

        Uses the original paralleliseOperationsWithBarriers / paralleliseOperations
        from the old pipeline via the routing adapter.
        """
        from qectostim.experiments.hardware_simulation.trapped_ion.routing import (
            paralleliseOperationsWithBarriers,
            paralleliseOperations,
        )

        def _old_schedule_to_time_map(schedule):
            """Convert old paralleliseOperations output to time → list-of-ops."""
            result = {}
            for t, par_op in schedule.items():
                ops = getattr(par_op, "operations", [par_op])
                result[t] = list(ops)
            return result

        metadata = circuit.metadata or {}
        old_ops = metadata.get("old_operations", [])
        barriers = metadata.get("old_barriers", metadata.get("barriers", []))

        if old_ops:
            # Use old scheduling on old operation objects
            if barriers:
                schedule = paralleliseOperationsWithBarriers(
                    old_ops, list(barriers),
                )
            else:
                schedule = paralleliseOperations(
                    old_ops,
                )
            time_map = _old_schedule_to_time_map(schedule)
            sched_name = "old_paralleliseOperations"
        else:
            # Fallback: empty schedule (no old ops available)
            time_map = {}
            sched_name = "empty"

        # Build scheduled operations from the time_map
        # Each entry: time → list of old operation objects
        scheduled_ops = []
        layers = []
        total_dur = 0.0
        for t in sorted(time_map.keys()):
            ops_at_t = time_map[t]
            layer_ops = []
            for op in ops_at_t:
                op_time = getattr(op, "operationTime", lambda: 0.0)()
                sched_op = ScheduledOperation(
                    operation=op,
                    start_time=t,
                    end_time=t + op_time,
                )
                scheduled_ops.append(sched_op)
                layer_ops.append(sched_op)
                end_t = t + op_time
                if end_t > total_dur:
                    total_dur = end_t
            if layer_ops:
                layers.append(layer_ops)

        return ScheduledCircuit(
            layers=layers,
            scheduled_ops=scheduled_ops,
            routed_circuit=circuit,
            batches={},
            total_duration=total_dur,
            metadata={
                "scheduler": sched_name,
                "num_barriers": len(barriers),
                "time_map": time_map,
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

    # -----------------------------------------------------------------
    # Logging configuration
    # -----------------------------------------------------------------

    @staticmethod
    def configure_logging(
        log_dir: str = "logs",
        log_prefix: str = "wise",
    ) -> None:
        """Set up file-based logging for the WISE route + SAT loggers.

        Creates ``<log_dir>/<prefix>_route.log`` and
        ``<log_dir>/<prefix>_sat.log`` with timestamped, per-PID
        formatting.  Idempotent — duplicate handlers are not added.

        Parameters
        ----------
        log_dir : str
            Directory for log files (created if absent).
        log_prefix : str
            Filename prefix (e.g. ``"wise_d3_k2"``).
        """
        os.makedirs(log_dir, exist_ok=True)
        fmt = logging.Formatter(
            "%(asctime)s [%(levelname)s] [pid=%(process)d] %(message)s"
        )

        for suffix, logger_name in [
            ("route", "wise.qccd.route"),
            ("sat", "wise.qccd.sat"),
        ]:
            logger = logging.getLogger(logger_name)
            logger.setLevel(logging.INFO)

            path = os.path.abspath(
                os.path.join(log_dir, f"{log_prefix}_{suffix}.log")
            )
            already = any(
                isinstance(h, logging.FileHandler)
                and getattr(h, "baseFilename", None) == path
                for h in logger.handlers
            )
            if not already:
                fh = logging.FileHandler(path)
                fh.setFormatter(fmt)
                logger.addHandler(fh)

    # -----------------------------------------------------------------
    # Hardware resource estimation (electrode / DAC counts)
    # -----------------------------------------------------------------

    # Electrode constants — from TABLE III, PRA 99, 022330
    NDE_LZ: int = 10   # DC electrodes per linear zone
    NDE_JZ: int = 20   # DC electrodes per junction zone
    NSE_Z: int = 10    # shim electrodes per zone

    @staticmethod
    def estimate_resources(
        num_ions: int,
        trap_capacity: int,
    ) -> Dict[str, int]:
        """Estimate DAC and electrode counts for a WISE grid.

        Parameters
        ----------
        num_ions : int
            Total ion slots across the grid (= m × n × k).
        trap_capacity : int
            Ions per trap segment (*k*).

        Returns
        -------
        dict
            ``{"electrodes": int, "dacs": int, "Njz": int, "Nlz": int}``
        """
        Njz = int(np.ceil(num_ions / trap_capacity))
        Nlz = num_ions - Njz

        Nde = WISECompiler.NDE_LZ * Nlz + WISECompiler.NDE_JZ * Njz
        Nse = WISECompiler.NSE_Z * (Njz + Nlz)

        electrodes = int(Nde + Nse)
        dacs = int(min(100, Nde) + np.ceil(Nse / 100))

        return {
            "electrodes": electrodes,
            "dacs": dacs,
            "Njz": Njz,
            "Nlz": Nlz,
        }

    # -----------------------------------------------------------------
    # Parallel config search  (ported from best_effort_compilation.py)
    # -----------------------------------------------------------------

    @staticmethod
    def search_configs(
        configs: List[Tuple[int, int, int, int]],
        *,
        d: int,
        m_traps: int,
        n_traps: int,
        trap_capacity: int = 2,
        barrier_threshold: float = np.inf,
        go_back_threshold: float = 0.0,
        base_pmax_in: Optional[int] = None,
        time_budget_s: Optional[float] = None,
        sat_workers_per_config: int = 4,
        max_total_workers: Optional[int] = None,
        verbose: bool = False,
        leaderboard_top_k: int = 5,
        gate_improvements: Sequence[float] = (1.0,),
        num_shots: int = 100_000,
    ) -> Tuple[Optional[Dict[str, Any]], List[Dict[str, Any]]]:
        """Search over routing configs in parallel, returning the best.

        Each config is a ``(lookahead, subgrid_width, subgrid_height,
        subgrid_increment)`` tuple.  For every config, the full WISE
        compilation pipeline is run in a worker process.

        Parameters
        ----------
        configs : list of (int, int, int, int)
            Routing parameter tuples to explore.
        d : int
            Surface-code distance.
        m_traps, n_traps : int
            Grid dimensions (columns, rows) of the WISE architecture.
        trap_capacity : int
            Ions per trap segment (*k*).
        barrier_threshold : float
            Disable barriers when ``trap_capacity > barrier_threshold``.
        go_back_threshold : float
            Routing go-back threshold.
        base_pmax_in : int or None
            Override for SAT ``pmax``.
        time_budget_s : float or None
            Global wall-clock time budget (seconds).
        sat_workers_per_config : int
            ``WISE_SAT_WORKERS`` env var set per worker.
        max_total_workers : int or None
            Process pool size (defaults to CPU count / sat_workers).
        verbose : bool
            Print live leaderboard.
        leaderboard_top_k : int
            How many top configs to display.
        gate_improvements : sequence of float
            Scaling factors for gate-error sweeps.
        num_shots : int
            Simulation shots per config.

        Returns
        -------
        (best_result, all_results) : (dict or None, list of dict)
        """
        if not configs:
            return None, []

        total_cpus = mp.cpu_count() or 1
        if max_total_workers is None:
            if sat_workers_per_config and sat_workers_per_config > 0:
                max_total_workers = max(1, total_cpus // sat_workers_per_config)
            else:
                max_total_workers = total_cpus
        max_total_workers = max(1, min(max_total_workers, len(configs)))

        start = time.time()
        all_results: List[Dict[str, Any]] = []
        best_result: Optional[Dict[str, Any]] = None

        def _is_better(a: Dict, b: Optional[Dict]) -> bool:
            if b is None:
                return True
            ea, eb = a["exec_time"], b["exec_time"]
            if np.isnan(ea):
                return False
            if np.isnan(eb):
                return True
            if ea < eb:
                return True
            if ea > eb:
                return False
            return a["comp_time"] < b["comp_time"]

        def _print_leaderboard() -> None:
            if not verbose or not all_results:
                return
            valid = [r for r in all_results if not np.isnan(r["exec_time"])]
            if not valid:
                return
            valid.sort(key=lambda r: (r["exec_time"], r["comp_time"]))
            top = valid[: max(1, leaderboard_top_k)]
            print("[WISE-SEARCH] Current top configs (by exec_time):")
            for r in top:
                print(
                    f"  la={r['lookahead']}, w={r['subgrid_width']}, "
                    f"h={r['subgrid_height']}, inc={r['subgrid_increment']}: "
                    f"exec={r['exec_time']:.3f}, comp={r['comp_time']:.1f}s"
                )
            print(flush=True)

        extra_kw: Dict[str, Any] = dict(
            d=d,
            m_traps=m_traps,
            n_traps=n_traps,
            trap_capacity=trap_capacity,
            barrier_threshold=barrier_threshold,
            go_back_threshold=go_back_threshold,
            base_pmax_in=base_pmax_in,
            sat_workers_per_config=sat_workers_per_config,
            time_budget_s=time_budget_s,
            num_shots=num_shots,
            gate_improvements=list(gate_improvements),
        )

        ctx = mp.get_context("spawn")
        executor = ProcessPoolExecutor(
            max_workers=max_total_workers, mp_context=ctx,
        )
        future_to_cfg: Dict[concurrent.futures.Future, Tuple] = {}

        try:
            for cfg in configs:
                fut = executor.submit(
                    _run_single_config_entry,
                    lookahead=cfg[0],
                    subgrid_width=cfg[1],
                    subgrid_height=cfg[2],
                    subgrid_increment=cfg[3],
                    **extra_kw,
                )
                future_to_cfg[fut] = cfg

            pending = set(future_to_cfg.keys())

            while pending:
                if time_budget_s is not None and (time.time() - start) >= time_budget_s:
                    if verbose:
                        print(
                            "[WISE-SEARCH] Global time budget reached; "
                            "stopping collection.",
                            flush=True,
                        )
                    break

                done, pending = concurrent.futures.wait(
                    pending, timeout=1.0,
                    return_when=concurrent.futures.FIRST_COMPLETED,
                )
                if not done:
                    continue

                for fut in done:
                    cfg = future_to_cfg[fut]
                    try:
                        res = fut.result()
                    except Exception as e:
                        if verbose:
                            print(
                                f"[WISE-SEARCH] crashed for cfg={cfg}: {e}",
                                flush=True,
                            )
                        res = _make_nan_result(
                            cfg, d, m_traps, n_traps, trap_capacity,
                            gate_improvements, error=repr(e),
                        )

                    all_results.append(res)
                    if _is_better(res, best_result):
                        best_result = res
                        if verbose:
                            print("[WISE-SEARCH] New best.", flush=True)
                            _print_leaderboard()

        except KeyboardInterrupt:
            if verbose:
                print("[WISE-SEARCH] KeyboardInterrupt; cancelling.", flush=True)
        finally:
            for f in future_to_cfg:
                if not f.done():
                    f.cancel()
            try:
                executor.shutdown(wait=False, cancel_futures=True)
            except Exception:
                pass

            # Aggressive worker teardown
            try:
                procs = getattr(executor, "_processes", None)
                if procs:
                    for pid, proc in list(procs.items()):
                        if proc.is_alive():
                            if verbose:
                                print(
                                    f"[WISE-SEARCH] Terminating worker PID {pid}",
                                    flush=True,
                                )
                            try:
                                proc.terminate()
                            except Exception:
                                pass
                    for _, proc in list(procs.items()):
                        if proc.is_alive():
                            try:
                                proc.join(timeout=1.0)
                            except Exception:
                                pass
                    for pid, proc in list(procs.items()):
                        if proc.is_alive():
                            if verbose:
                                print(
                                    f"[WISE-SEARCH] Killing worker PID {pid}",
                                    flush=True,
                                )
                            try:
                                proc.kill()
                            except Exception:
                                pass
            except Exception:
                pass

        if verbose:
            elapsed = time.time() - start
            print(
                f"[WISE-SEARCH] Done in {elapsed:.1f}s; "
                f"{len(all_results)} configs completed.",
                flush=True,
            )
            _print_leaderboard()

        return best_result, all_results


# =====================================================================
# Module-level helpers for multiprocessing (must be pickle-able)
# =====================================================================

def _make_nan_result(
    cfg: Tuple[int, int, int, int],
    d: int,
    m_traps: int,
    n_traps: int,
    trap_capacity: int,
    gate_improvements: Sequence[float],
    error: str = "",
) -> Dict[str, Any]:
    """Create an all-NaN result dict for a failed config."""
    return {
        "lookahead": cfg[0],
        "subgrid_width": cfg[1],
        "subgrid_height": cfg[2],
        "subgrid_increment": cfg[3],
        "d": d,
        "m_traps": m_traps,
        "n_traps": n_traps,
        "trap_capacity": trap_capacity,
        "exec_time": float("nan"),
        "comp_time": 0.0,
        "reconfigTime": float("nan"),
        "ElapsedTime": float("nan"),
        "Operations": float("nan"),
        "MeanConcurrency": float("nan"),
        "QubitOperations": float("nan"),
        "LogicalErrorRates": [float("nan") for _ in gate_improvements],
        "PhysicalZErrorRates": [float("nan") for _ in gate_improvements],
        "PhysicalXErrorRates": [float("nan") for _ in gate_improvements],
        "error": error,
    }


def _run_single_config_entry(
    lookahead: int,
    subgrid_width: int,
    subgrid_height: int,
    subgrid_increment: int,
    *,
    d: int,
    m_traps: int,
    n_traps: int,
    trap_capacity: int,
    barrier_threshold: float,
    go_back_threshold: float,
    base_pmax_in: Optional[int],
    sat_workers_per_config: int,
    time_budget_s: Optional[float],
    gate_improvements: Sequence[float] = (1.0,),
    num_shots: int = 100_000,
) -> Dict[str, Any]:
    """Worker entry point — runs one config in a sub-process.

    Sets SAT worker / timeout env-vars, then calls ``run_single_config``.
    """
    WISECompiler.configure_logging(
        log_prefix=f"wise_d{d}_k{trap_capacity}",
    )

    if sat_workers_per_config and sat_workers_per_config > 0:
        os.environ["WISE_SAT_WORKERS"] = str(sat_workers_per_config)

    if time_budget_s is not None and time_budget_s > 0:
        cap = max(time_budget_s / 2.0, 1.0)
        for var in ("WISE_MAX_SAT_TIME", "WISE_MAX_RC2_TIME"):
            cur = os.environ.get(var)
            val = cap
            if cur is not None:
                try:
                    val = min(cap, float(cur)) if float(cur) > 0 else cap
                except ValueError:
                    pass
            os.environ[var] = f"{val}"

    exec_time, comp_time, results, reconfig_time = run_single_config(
        lookahead=lookahead,
        subgrid_width=subgrid_width,
        subgrid_height=subgrid_height,
        subgrid_increment=subgrid_increment,
        d=d,
        m_traps=m_traps,
        n_traps=n_traps,
        trap_capacity=trap_capacity,
        barrier_threshold=barrier_threshold,
        go_back_threshold=go_back_threshold,
        base_pmax_in=base_pmax_in,
        gate_improvements=gate_improvements,
        num_shots=num_shots,
    )

    row: Dict[str, Any] = {
        "lookahead": lookahead,
        "subgrid_width": subgrid_width,
        "subgrid_height": subgrid_height,
        "subgrid_increment": subgrid_increment,
        "d": d,
        "m_traps": m_traps,
        "n_traps": n_traps,
        "trap_capacity": trap_capacity,
        "exec_time": exec_time,
        "comp_time": comp_time,
        "reconfigTime": reconfig_time,
    }
    row.update(results)
    return row


def run_single_config(
    lookahead: int,
    subgrid_width: int,
    subgrid_height: int,
    subgrid_increment: int,
    *,
    d: int = 4,
    m_traps: int = 6,
    n_traps: int = 6,
    trap_capacity: int = 2,
    barrier_threshold: float = np.inf,
    go_back_threshold: float = 0.0,
    base_pmax_in: Optional[int] = None,
    gate_improvements: Sequence[float] = (1.0,),
    num_shots: int = 100_000,
) -> Tuple[float, float, Dict[str, Any], float]:
    """Run WISE routing for a single parameter configuration.

    Uses the full WISECompiler pipeline: decompose → map → route → schedule.
    Collects execution time, operation counts, and concurrency metrics.

    Parameters
    ----------
    lookahead : int
        SAT solver look-ahead depth.
    subgrid_width, subgrid_height, subgrid_increment : int
        Patch-slicer dimensions.
    d : int
        Surface-code distance.
    m_traps, n_traps : int
        WISE grid dimensions (columns, rows).
    trap_capacity : int
        Ions per trap (*k*).
    barrier_threshold : float
        Disable barriers when capacity > threshold.
    gate_improvements : sequence of float
        Gate-error scaling factors for sweep.
    num_shots : int
        Simulation shots.

    Returns
    -------
    (exec_time, comp_time, results, reconfig_time)
    """
    import stim
    from qectostim.experiments.hardware_simulation.trapped_ion.architecture import (
        WISEArchitecture,
    )
    from qectostim.experiments.hardware_simulation.trapped_ion.routing.config import (
        WISERoutingConfig,
    )

    t0 = time.perf_counter()
    try:
        # 1) Build stim surface-code circuit
        circuit = stim.Circuit.generated(
            "surface_code:rotated_memory_z",
            rounds=1,
            distance=d,
        )
        num_ions = m_traps * n_traps * trap_capacity

        # 2) Build architecture + routing config
        arch = WISEArchitecture(
            col_groups=m_traps,
            rows=n_traps,
            ions_per_segment=trap_capacity,
        )
        rc = WISERoutingConfig(
            subgridsize=(subgrid_width, subgrid_height, subgrid_increment),
            lookahead=lookahead,
        )

        # 3) Build compiler and run pipeline
        compiler = WISECompiler(architecture=arch, routing_config=rc)
        native = compiler.decompose_to_native(circuit)
        mapped = compiler.map_qubits(native)
        routed = compiler.route(mapped)
        scheduled = compiler.schedule(routed)

        # 4) Extract metrics from scheduled result
        # Old pipeline uses: exec_time = float(max(parallelOpsMap.keys()))
        # i.e. the max START time of the last batch (in seconds).
        # New pipeline time_map keys are in µs → convert to seconds.
        sched_meta = scheduled.metadata or {}
        time_map_raw = sched_meta.get("time_map", {})
        if time_map_raw:
            exec_time = float(max(time_map_raw.keys())) * 1e-6
        else:
            exec_time = 0.0
        reconfig_time = (routed.metadata or {}).get("reconfig_time", 0.0)

        # Count operations and concurrency
        routing_meta = routed.metadata or {}
        num_old_ops = routing_meta.get("num_old_ops", len(routed.operations))
        num_qubit_ops = len(mapped.native_circuit.operations) if mapped.native_circuit else 0

        time_map = sched_meta.get("time_map", {})
        mean_concurrency = 0.0
        if time_map:
            slot_sizes = [
                len(ops) if isinstance(ops, (list, tuple)) else 1
                for ops in time_map.values()
            ]
            mean_concurrency = float(np.mean(slot_sizes)) if slot_sizes else 0.0

        t1 = time.perf_counter()
        comp_time = t1 - t0

        # 5) Simulation / error rates  (placeholder — returns NaN until
        #    TrappedIonSimulator integration is wired in)
        logical_errors: List[float] = [float("nan") for _ in gate_improvements]
        physical_z: List[float] = [float("nan") for _ in gate_improvements]
        physical_x: List[float] = [float("nan") for _ in gate_improvements]

        results: Dict[str, Any] = {
            "ElapsedTime": exec_time,
            "Operations": num_old_ops,
            "MeanConcurrency": mean_concurrency,
            "QubitOperations": num_qubit_ops,
            "LogicalErrorRates": logical_errors,
            "PhysicalZErrorRates": physical_z,
            "PhysicalXErrorRates": physical_x,
        }

        hw = WISECompiler.estimate_resources(num_ions, trap_capacity)
        results["DACs"] = hw["dacs"]
        results["Electrodes"] = hw["electrodes"]

        return exec_time, comp_time, results, reconfig_time

    except Exception as e:
        t1 = time.perf_counter()
        comp_time = t1 - t0
        print(
            f"[WARN] Failed: la={lookahead}, "
            f"sub=({subgrid_width},{subgrid_height},{subgrid_increment}), "
            f"d={d}, m={m_traps}, n={n_traps}: {e!r}"
        )
        results = {
            "ElapsedTime": float("nan"),
            "Operations": float("nan"),
            "MeanConcurrency": float("nan"),
            "QubitOperations": float("nan"),
            "LogicalErrorRates": [float("nan") for _ in gate_improvements],
            "PhysicalZErrorRates": [float("nan") for _ in gate_improvements],
            "PhysicalXErrorRates": [float("nan") for _ in gate_improvements],
        }
        return float("nan"), comp_time, results, float("nan")