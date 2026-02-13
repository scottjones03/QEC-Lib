# src/qectostim/experiments/hardware_simulation/trapped_ion/compilers/networked.py
"""
Compiler for the Networked Grid QCCD architecture.

Uses heuristic clustering for qubit mapping and junction-based greedy
routing via the fully-connected junction topology.
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


class NetworkedGridCompiler(QCCDCompiler):
    """Compiler for the Networked Grid architecture.

    Inherits gate decomposition, routing, and scheduling from
    :class:`QCCDCompiler`.  Overrides ``map_qubits`` to use the
    simple cluster arrangement algorithm (no hill-climbing needed
    for a 1-D linear arrangement of traps).
    """

    def __init__(
        self,
        architecture: "TrappedIonArchitecture",
        optimization_level: int = 1,
    ):
        super().__init__(architecture, optimization_level, use_wise_routing=False)

    def map_qubits(self, circuit: NativeCircuit) -> MappedCircuit:
        """Map logical qubits to physical ions via clustering.

        Falls back to sequential mapping if the architecture is not a
        :class:`NetworkedGridArchitecture`.
        """
        from qectostim.experiments.hardware_simulation.trapped_ion.architecture import (
            NetworkedGridArchitecture,
            Ion,
        )
        from qectostim.experiments.hardware_simulation.trapped_ion.clustering import (
            regular_partition,
            arrange_clusters,
        )

        arch = self.architecture
        if not isinstance(arch, NetworkedGridArchitecture):
            return super().map_qubits(circuit)

        all_ions = arch.qccd_graph.ions
        ion_list = sorted(all_ions.values(), key=lambda i: i.idx)

        measurement_ions = [i for i in ion_list if i.position[0] % 2 == 0]
        data_ions = [i for i in ion_list if i.position[0] % 2 != 0]
        if not data_ions:
            data_ions = ion_list[len(ion_list) // 2:]
            measurement_ions = ion_list[: len(ion_list) // 2]

        clusters = regular_partition(
            measurement_ions, data_ions, arch.ions_per_trap,
        )

        # 1-D grid positions: (0, row) for each trap
        all_grid_pos = [(0, row) for row in range(arch.num_traps)]

        if clusters and all_grid_pos:
            grid_positions = arrange_clusters(clusters, all_grid_pos)
        else:
            grid_positions = []

        mapping = QubitMapping()
        mapped_ions = set()
        for cidx, (_, row) in enumerate(grid_positions):
            if cidx < len(clusters):
                for ion in clusters[cidx][0]:
                    if ion.idx not in mapped_ions:
                        mapping.assign(ion.idx, ion.idx)
                        mapped_ions.add(ion.idx)

        for logical_q in range(circuit.num_qubits):
            if logical_q not in mapped_ions:
                mapping.assign(logical_q, logical_q)

        return MappedCircuit(
            native_circuit=circuit,
            mapping=mapping,
            metadata={"mapping_strategy": "networked_grid_clustering"},
        )
