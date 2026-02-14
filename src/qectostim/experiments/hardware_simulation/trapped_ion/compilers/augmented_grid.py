# src/qectostim/experiments/hardware_simulation/trapped_ion/compilers/augmented_grid.py
"""
Compiler for the Augmented Grid QCCD architecture.

Uses heuristic clustering (``regular_partition`` + ``hill_climb_on_arrange_clusters``)
for qubit mapping and junction-based greedy routing.
"""
from __future__ import annotations

import logging
from typing import (
    Any,
    Dict,
    List,
    TYPE_CHECKING,
)

from qectostim.experiments.hardware_simulation.core.pipeline import (
    NativeCircuit,
    MappedCircuit,
    RoutedCircuit,
    ScheduledCircuit,
    QubitMapping,
)
from qectostim.experiments.hardware_simulation.trapped_ion.compilers.qccd import (
    QCCDCompiler,
)

if TYPE_CHECKING:
    from qectostim.experiments.hardware_simulation.trapped_ion.architecture import (
        TrappedIonArchitecture,
    )


_logger = logging.getLogger(__name__)


class AugmentedGridCompiler(QCCDCompiler):
    """Compiler for the Augmented Grid architecture.

    Inherits gate decomposition, routing, and scheduling from
    :class:`QCCDCompiler`.  Overrides ``map_qubits`` to use the
    Hungarian-matching cluster placement algorithm from
    :mod:`..clustering`.
    """

    def __init__(
        self,
        architecture: "TrappedIonArchitecture",
        optimization_level: int = 1,
    ):
        super().__init__(architecture, optimization_level, use_wise_routing=False)

    def map_qubits(self, circuit: NativeCircuit) -> MappedCircuit:
        """Map logical qubits to physical ions via clustering + Hungarian placement.

        Replicates the old ``processCircuitAugmentedGrid`` pipeline:

        1. Create ``Ion`` objects with proper **D** / **M** labels from
           the circuit's ``QUBIT_COORDS``.
        2. Cluster with ``regularPartition(isWISEArch=False)`` so each
           trap keeps one routing-headroom slot free.
        3. Place clusters on the grid via ``hillClimbOnArrangeClusters``.
        4. Rebuild the architecture's traps with the clustered ions.
        5. Return a mapping from logical qubit index → physical ion
           position in the new ``qubit_ions`` list.

        Falls back to the parent ``QCCDCompiler.map_qubits`` if the
        architecture is not an :class:`AugmentedGridArchitecture`.
        """
        from qectostim.experiments.hardware_simulation.trapped_ion.architecture import (
            AugmentedGridArchitecture,
            Ion,
        )
        from qectostim.experiments.hardware_simulation.trapped_ion.routing import (
            regularPartition as regular_partition,
            hillClimbOnArrangeClusters as hill_climb_on_arrange_clusters,
        )

        arch = self.architecture
        if not isinstance(arch, AugmentedGridArchitecture):
            return super().map_qubits(circuit)

        # ── 1. Derive D/M ions from the circuit ─────────────────────────
        # Extract QUBIT_COORDS from the underlying stim circuit.
        stim_src = getattr(circuit, "stim_source", None) or circuit
        qubit_coords: Dict[int, tuple] = {}
        if hasattr(stim_src, "flattened"):
            for inst in stim_src.flattened():
                if inst.name == "QUBIT_COORDS":
                    args = inst.gate_args_copy()
                    for t in inst.targets_copy():
                        qubit_coords[t.value] = tuple(args)

        # Determine which logical qubits are data vs measurement.
        # The old code uses coordinate parity: x-coord odd → Data,
        # x-coord even → Measurement.  We also accept an explicit
        # data_qubit_indices from QEC metadata when available.
        n_logical = circuit.num_qubits
        meta = getattr(circuit, "qec_metadata", None)
        if meta and hasattr(meta, "data_qubit_indices") and meta.data_qubit_indices:
            data_set = set(meta.data_qubit_indices)
        else:
            # Fallback: derive from QUBIT_COORDS parity (old convention)
            data_set = set()
            for q in range(n_logical):
                coords = qubit_coords.get(q)
                if coords is not None:
                    # Old convention: odd x-coordinate → data qubit
                    if int(coords[0]) % 2 != 0:
                        data_set.add(q)
                else:
                    # No coords — treat first n qubits as data
                    # (rotated surface code: n_data = d*d)
                    pass
            if not data_set:
                # Last resort: assume first half are data
                import math
                d = int(math.sqrt(n_logical))
                n_data_guess = d * d if d * d < n_logical else n_logical // 2
                data_set = set(range(n_data_guess))

        data_ions: List[Ion] = []
        measurement_ions: List[Ion] = []
        logical_to_ion: Dict[int, Ion] = {}

        for q in range(n_logical):
            coords = qubit_coords.get(q, (0.0, 0.0))
            if q in data_set:
                ion = Ion(idx=q, label="D", position=coords)
                data_ions.append(ion)
            else:
                ion = Ion(idx=q, label="M", position=coords)
                measurement_ions.append(ion)
            logical_to_ion[q] = ion

        # ── 2. Cluster with routing headroom ─────────────────────────────
        all_grid_pos = list(arch.traps_dict.keys())
        clusters = regular_partition(
            measurement_ions, data_ions, arch.ions_per_trap,
            maxClusters=len(all_grid_pos),
            isWISEArch=False,   # AugGrid: eff_capacity = k-1
        )

        # ── 3. Place clusters on the grid ────────────────────────────────
        if clusters and all_grid_pos:
            grid_positions = hill_climb_on_arrange_clusters(
                clusters, all_grid_pos,
            )
        else:
            grid_positions = []

        # ── 4. Build initial_ions dict and rebuild architecture ──────────
        trap_for_grid: Dict[tuple, List[Ion]] = {}
        for cidx, grid_pos in enumerate(grid_positions):
            if cidx < len(clusters):
                gkey = (int(grid_pos[0]), int(grid_pos[1]))
                trap_for_grid[gkey] = list(clusters[cidx][0])

        # Rebuild the architecture's topology with the clustered ions.
        # _build_grid_topology resets _qccd_graph, _traps_dict, etc.
        arch._build_grid_topology(initial_ions=trap_for_grid)

        # ── 5. Build logical → physical mapping ─────────────────────────
        # After rebuild, qubit_ions is the new ordered list.
        # Walk the traps to find each ion and map logical→physical.
        mapping = QubitMapping()
        qi = arch.qccd_graph.qubit_ions  # new ordered list
        ion_obj_to_phys = {id(ion): pidx for pidx, ion in enumerate(qi)}

        for lq, ion in logical_to_ion.items():
            phys = ion_obj_to_phys.get(id(ion))
            if phys is not None:
                mapping.assign(lq, phys)

        return MappedCircuit(
            native_circuit=circuit,
            mapping=mapping,
            metadata={"mapping_strategy": "augmented_grid_clustering"},
        )
