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

        Falls back to sequential mapping if the architecture is not an
        :class:`AugmentedGridArchitecture`.
        """
        from qectostim.experiments.hardware_simulation.trapped_ion.architecture import (
            AugmentedGridArchitecture,
            Ion,
        )
        from qectostim.experiments.hardware_simulation.trapped_ion.clustering import (
            regular_partition,
            hill_climb_on_arrange_clusters,
        )

        arch = self.architecture
        if not isinstance(arch, AugmentedGridArchitecture):
            return super().map_qubits(circuit)

        # Build ions for each logical qubit using positions from the architecture.
        # Ions sharing a trap have identical coordinates; give each a unique
        # micro-offset (< 0.01) so the BSP clustering can distinguish them.
        all_ions = arch.qccd_graph.ions
        ion_list = sorted(all_ions.values(), key=lambda i: i.idx)

        _seen_pos: Dict[tuple, int] = {}
        for ion in ion_list:
            key = tuple(ion.position)
            slot = _seen_pos.get(key, 0)
            _seen_pos[key] = slot + 1
            if slot > 0:
                eps = slot * 0.001
                ion.position = (ion.position[0] + eps, ion.position[1] + eps)

        # Split into data (first n_logical) and measurement (remaining).
        n_logical = circuit.num_qubits
        data_ions = ion_list[:n_logical]
        measurement_ions = ion_list[n_logical:]

        # Cluster — limit to # available traps so placement can succeed
        all_grid_pos = list(arch.traps_dict.keys())
        clusters = regular_partition(
            measurement_ions, data_ions, arch.ions_per_trap,
            max_clusters=len(all_grid_pos),
        )

        # Build available grid positions from the architecture traps
        all_grid_pos = list(arch.traps_dict.keys())

        # Place clusters on grid
        if clusters and all_grid_pos:
            grid_positions = hill_climb_on_arrange_clusters(
                clusters, all_grid_pos,
            )
        else:
            grid_positions = []

        # Build mapping: cluster_idx → grid_pos → physical trap → ion indices
        mapping = QubitMapping()
        mapped_ions = set()
        for cidx, (col, row) in enumerate(grid_positions):
            if cidx < len(clusters):
                cluster_ions = clusters[cidx][0]
                for ion in cluster_ions:
                    if ion.idx not in mapped_ions:
                        mapping.assign(ion.idx, ion.idx)
                        mapped_ions.add(ion.idx)

        # Map any remaining
        for logical_q in range(circuit.num_qubits):
            if logical_q not in mapped_ions:
                mapping.assign(logical_q, logical_q)

        return MappedCircuit(
            native_circuit=circuit,
            mapping=mapping,
            metadata={"mapping_strategy": "augmented_grid_clustering"},
        )
