"""
Architecture subclasses for QCCD trapped-ion hardware.

Provides three concrete topologies that extend :class:`QCCDArch`:

- :class:`AugmentedGridArchitecture` — augmented grid with diagonal traps
- :class:`WISEArchitecture` — WISE (Wired Ion Surface Electrode) topology
- :class:`NetworkedGridArchitecture` — fully-connected grid via junction chain
"""

from typing import (
    Sequence,
    List,
    Tuple,
    Optional,
    Dict,
)

import numpy as np

from .qccd_arch import QCCDArch
from .qccd_nodes import (
    Ion,
    QubitIon,
    SpectatorIon,
    Trap,
    ManipulationTrap,
    Junction,
    QCCDWiseArch,
)
from ..compiler.qccd_qubits_to_ions import (
    regularPartition,
    hillClimbOnArrangeClusters,
    arrangeClusters,
)


# ============================================================================
# AugmentedGridArchitecture
# ============================================================================


class AugmentedGridArchitecture(QCCDArch):
    """QCCD topology arranged as an augmented grid.

    Even rows hold main traps at ``(2c, 2r)``; odd rows add diagonal
    traps at ``(2c+1, 2r+1)``.  Junctions connect vertically between
    even rows, with horizontal edges linking diagonal traps to junctions.

    Parameters
    ----------
    trap_capacity : int
        Number of ion slots per trap.
    rows, cols : int
        Grid dimensions (before padding).
    padding : int
        Padding rows/columns added around the grid.
    """

    # Hill-climb parameters for cluster placement
    start_score: int = 1
    score_delta: int = 2
    joinDisjointClusters: bool = False
    minIters: int = 100
    maxIters: int = 1000

    def __init__(
        self,
        trap_capacity: int = 2,
        rows: int = 1,
        cols: int = 5,
        padding: int = 1,
    ):
        super().__init__()
        self.trap_capacity = trap_capacity
        self.rows = rows
        self.cols = cols
        self.padding = padding

    def build_topology(
        self,
        measurement_ions: Sequence[Ion],
        data_ions: Sequence[Ion],
        ion_mapping: Dict[int, Tuple[Ion, Tuple[int, int]]],
    ) -> Tuple[Sequence, Sequence[int]]:
        """Build the augmented grid topology and return parsed instructions context.

        This mirrors the logic previously in
        ``QCCDCircuit.processCircuitAugmentedGrid``.

        Parameters
        ----------
        measurement_ions : Sequence[Ion]
            Measurement (ancilla) ions.
        data_ions : Sequence[Ion]
            Data ions.
        ion_mapping : Dict[int, Tuple[Ion, Tuple[int, int]]]
            ``{stim_idx: (Ion, coords)}`` mapping.

        Returns
        -------
        None
            Topology is built in-place on ``self``.

        Raises
        ------
        ValueError
            If there are not enough traps for the ions.
        """
        trapCapacity = self.trap_capacity
        if (trapCapacity - 1) * (
            (self.rows - 1) * (2 * self.cols - 1) + self.cols
        ) < len(ion_mapping):
            raise ValueError("processCircuit: not enough traps")

        clusters = regularPartition(
            measurement_ions, data_ions, trapCapacity
        )

        cs, rs = self.cols, self.rows
        allGridPos = []
        for r in range(rs):
            for c in range(cs):
                allGridPos.append((2 * c, 2 * r))
                if c < cs - 1 and r < rs - 1:
                    allGridPos.append((2 * c + 1, 2 * r + 1))

        gridPositions = hillClimbOnArrangeClusters(
            clusters, allGridPos=allGridPos
        )
        gridPositions = [
            (c + self.padding, r + self.padding) for (c, r) in gridPositions
        ]
        rows = self.rows + 2 * self.padding
        cols = self.cols + 2 * self.padding
        trap_for_grid = {
            (col, row): clusters[trapIdx]
            for trapIdx, (col, row) in enumerate(gridPositions)
        }
        self._originalArrangement = {}

        traps_dict: Dict[Tuple[int, int], Trap] = {}
        for row in range(rows):
            for col in range(cols):
                if (2 * col, 2 * row) in trap_for_grid:
                    ions = trap_for_grid[(2 * col, 2 * row)][0]
                else:
                    ions = []
                traps_dict[(2 * col, 2 * row)] = self.addManipulationTrap(
                    *self._gridToCoordinate((2 * col, 2 * row), trapCapacity),
                    ions,
                    isHorizontal=(rows == 1),
                    capacity=trapCapacity,
                )
                self._originalArrangement[traps_dict[(2 * col, 2 * row)]] = ions

            if row == rows - 1:
                break

            for col in range(cols - 1):
                if (2 * col + 1, 2 * row + 1) in trap_for_grid:
                    ions = trap_for_grid[(2 * col + 1, 2 * row + 1)][0]
                else:
                    ions = []
                traps_dict[(2 * col + 1, 2 * row + 1)] = self.addManipulationTrap(
                    *self._gridToCoordinate(
                        (2 * col + 1, 2 * row + 1), trapCapacity
                    ),
                    ions,
                    isHorizontal=True,
                    capacity=trapCapacity,
                )
                self._originalArrangement[
                    traps_dict[(2 * col + 1, 2 * row + 1)]
                ] = ions

        if rows == 1:
            for (col, r), trap_node in traps_dict.items():
                if (col + 2, r) in traps_dict:
                    self.addEdge(trap_node, traps_dict[(col + 2, r)])
        else:
            junctions_dict: Dict[Tuple[int, int], Junction] = {}
            for (col, row), trap_node in traps_dict.items():
                # Add vertical edges (between even rows)
                if col % 2 == 0 and (col, row + 2) in traps_dict:
                    junction = self.addJunction(
                        *(
                            (
                                self._gridToCoordinate((col, row), trapCapacity)
                                + self._gridToCoordinate(
                                    (col, row + 2), trapCapacity
                                )
                            )
                            / 2
                        ),
                    )
                    junctions_dict[(col, row + 1)] = junction
                    self.addEdge(trap_node, junction)
                    self.addEdge(junction, traps_dict[(col, row + 2)])

            # Add horizontal edges between traps and junctions in the same row
            for row in range(rows - 1):
                for col in range(cols - 1):
                    if (2 * col, 2 * row + 1) in junctions_dict and (
                        2 * col + 1,
                        2 * row + 1,
                    ) in traps_dict:
                        self.addEdge(
                            junctions_dict[(2 * col, 2 * row + 1)],
                            traps_dict[(2 * col + 1, 2 * row + 1)],
                        )
                    if (2 * col + 1, 2 * row + 1) in traps_dict and (
                        2 * col + 2,
                        2 * row + 1,
                    ) in junctions_dict:
                        self.addEdge(
                            traps_dict[(2 * col + 1, 2 * row + 1)],
                            junctions_dict[(2 * col + 2, 2 * row + 1)],
                        )

        if any(i.parent is None for i in self.ions.values()):
            raise ValueError(
                f"Ions not in traps for {trapCapacity}"
                f" and {len(measurement_ions) + len(data_ions)}"
            )


# ============================================================================
# WISEArchitecture
# ============================================================================


class WISEArchitecture(QCCDArch):
    """QCCD topology using the WISE (Wired Ion Surface Electrode) layout.

    Traps are placed in paired columns ``(2c, r)`` with junction columns
    at odd x-positions.  Spectator ions can be added as placeholders.

    Parameters
    ----------
    wise_config : QCCDWiseArch
        Configuration with ``m`` (columns), ``n`` (rows), ``k``
        (ions per trap).
    add_spectators : bool
        Whether to add spectator ions as placeholders.
    compact_clustering : bool
        Whether to use compact clustering.
    """

    # Hill-climb parameters
    start_score: int = 1
    score_delta: int = 2
    joinDisjointClusters: bool = False
    minIters: int = 100
    maxIters: int = 1000

    def __init__(
        self,
        wise_config: QCCDWiseArch,
        add_spectators: bool = True,
        compact_clustering: bool = True,
    ):
        super().__init__()
        self.wise_config = wise_config
        self.add_spectators = add_spectators
        self.compact_clustering = compact_clustering

    def build_topology(
        self,
        measurement_ions: Sequence[Ion],
        data_ions: Sequence[Ion],
        ion_mapping: Dict[int, Tuple[Ion, Tuple[int, int]]],
    ) -> None:
        """Build the WISE topology.

        Mirrors the logic previously in
        ``QCCDCircuit.processCircuitWiseArch``.

        Parameters
        ----------
        measurement_ions : Sequence[Ion]
            Measurement (ancilla) ions.
        data_ions : Sequence[Ion]
            Data ions.
        ion_mapping : Dict[int, Tuple[Ion, Tuple[int, int]]]
            ``{stim_idx: (Ion, coords)}`` mapping.  May be mutated
            to add spectator ions.
        """
        wiseArch = self.wise_config
        if self.compact_clustering and (
            wiseArch.m * wiseArch.n * wiseArch.k < len(ion_mapping)
        ):
            raise ValueError("processCircuit: not enough traps")

        clusters = regularPartition(
            measurement_ions,
            data_ions,
            wiseArch.k,
            isWISEArch=self.compact_clustering,
            maxClusters=(
                wiseArch.m * wiseArch.n if self.compact_clustering else None
            ),
        )

        cs, rs = wiseArch.m, wiseArch.n
        allGridPos = []
        for r in range(rs):
            for c in range(cs):
                allGridPos.append((c, r))
        gridPositions = hillClimbOnArrangeClusters(
            clusters, allGridPos=allGridPos
        )
        gridPositions = [(c, r) for (c, r) in gridPositions]
        rows = wiseArch.n
        cols = wiseArch.m
        trap_for_grid = {
            (2 * col, row): clusters[trapIdx]
            for trapIdx, (col, row) in enumerate(gridPositions)
        }
        self._originalArrangement = {}

        traps_dict: Dict[Tuple[int, int], Trap] = {}
        for row in range(rows):
            for col in range(cols):
                if (2 * col, row) in trap_for_grid:
                    ions = trap_for_grid[(2 * col, row)][0]
                else:
                    ions = []
                if self.add_spectators:
                    maxIdx = max(ion_mapping.keys())
                    nplaceholds = wiseArch.k - len(ions)
                    for i in range(nplaceholds):
                        ion = SpectatorIon("#bbbbbb", "P")
                        idx = maxIdx + 1 + i
                        ion.set(idx, *ion.pos)
                        ion_mapping[idx] = ion, ion.pos
                        ions.append(ion)
                traps_dict[(2 * col, row)] = self.addManipulationTrap(
                    *self._gridToCoordinate((2 * col, row), wiseArch.k),
                    ions,
                    isHorizontal=True,
                    capacity=wiseArch.k,
                )
                self._originalArrangement[traps_dict[(2 * col, row)]] = ions

        if rows == 1:
            for (col, r), trap_node in traps_dict.items():
                if (col + 2, r) in traps_dict:
                    self.addEdge(trap_node, traps_dict[(col + 2, r)])
        else:
            junctions_dict: Dict[Tuple[int, int], Junction] = {}
            for (col, row), trap_node in traps_dict.items():
                if (col, row + 1) in traps_dict:
                    if (col + 1, row) not in junctions_dict:
                        junctionTop = self.addJunction(
                            *self._gridToCoordinate(
                                (col + 1, row), wiseArch.k
                            ),
                        )
                        junctions_dict[(col + 1, row)] = junctionTop
                    if (col + 1, row + 1) not in junctions_dict:
                        junctionBottom = self.addJunction(
                            *self._gridToCoordinate(
                                (col + 1, row + 1), wiseArch.k
                            ),
                        )
                        junctions_dict[(col + 1, row + 1)] = junctionBottom

            for (col, row), junctionTop in junctions_dict.items():
                if (col, row + 1) in junctions_dict:
                    self.addEdge(
                        junctions_dict[(col, row + 1)], junctionTop
                    )

            # Horizontal edges between traps and junctions in the same row
            for row in range(rows):
                for col in range(cols):
                    if (2 * col + 1, row) in junctions_dict and (
                        2 * col,
                        row,
                    ) in traps_dict:
                        self.addEdge(
                            junctions_dict[(2 * col + 1, row)],
                            traps_dict[(2 * col, row)],
                        )
                    if (2 * col + 2, row) in traps_dict and (
                        2 * col + 1,
                        row,
                    ) in junctions_dict:
                        self.addEdge(
                            traps_dict[(2 * col + 2, row)],
                            junctions_dict[(2 * col + 1, row)],
                        )

        if any(i.parent is None for i in self.ions.values()):
            raise ValueError(
                f"Ions not in traps for {wiseArch.k}"
                f" and {len(measurement_ions) + len(data_ions)}"
            )

    # ------------------------------------------------------------------ #
    # Per-block build_topology for gadget compilation (§3.1.1)
    # ------------------------------------------------------------------ #

    def build_topology_per_block(
        self,
        block_sub_grids: Dict[str, "BlockSubGrid"],
        measurement_ions_per_block: Dict[str, List[Ion]],
        data_ions_per_block: Dict[str, List[Ion]],
        ion_mapping: Dict[int, "Tuple[Ion, Tuple[int, int]]"],
    ) -> None:
        """Build WISE topology with per-block qubit-to-ion mapping.

        Instead of running ``regularPartition`` globally (which
        interleaves block ions across the grid), this runs independent
        ``regularPartition`` + ``hillClimbOnArrangeClusters`` for each
        block on its own disjoint sub-grid region.

        The result is that no two blocks share any trap, enabling Level 1
        spatial slicing for parallel per-block EC routing.

        Parameters
        ----------
        block_sub_grids : Dict[str, BlockSubGrid]
            Per-block sub-grid allocations with ``grid_region``.
        measurement_ions_per_block : Dict[str, List[Ion]]
            Measurement (ancilla) ions per block.
        data_ions_per_block : Dict[str, List[Ion]]
            Data ions per block.
        ion_mapping : Dict[int, Tuple[Ion, Tuple[int, int]]]
            ``{stim_idx: (Ion, coords)}``.  May be mutated to add
            spectator ions.
        """
        wiseArch = self.wise_config

        # --- Per-block ion placement ---
        # Build a global trap_for_grid with each block's ions placed
        # into their own sub-grid region.
        #
        # When a block has a ``coord_to_trap_pos`` mapping (coordinate-
        # aware grid sizing), use deterministic placement that mirrors
        # the stim circuit's spatial layout exactly.  Otherwise fall
        # back to stochastic regularPartition + hillClimb.
        trap_for_grid = {}

        for block_name, sg in block_sub_grids.items():
            r0, c0, r1, c1 = sg.grid_region
            block_rows = r1 - r0
            block_cols = c1 - c0

            m_ions = measurement_ions_per_block.get(block_name, [])
            d_ions = data_ions_per_block.get(block_name, [])

            if not m_ions and not d_ions:
                continue

            # -----------------------------------------------------------
            # Deterministic coordinate-based placement (Option B)
            # -----------------------------------------------------------
            if sg.coord_to_trap_pos:
                # Build a layout grid: (row, trap_col) → list of ions
                _trap_ions: Dict[Tuple[int, int], list] = {}
                _unplaced: list = []

                # Process data ions first so code qubits claim
                # their deterministic positions before bridge ancillas.
                for ion in list(d_ions) + list(m_ions):
                    pos_key = (int(ion.pos[0]), int(ion.pos[1]))
                    placement = sg.coord_to_trap_pos.get(pos_key)
                    if placement is not None:
                        row, trap_col, _slot = placement
                        key = (row, trap_col)
                        if key not in _trap_ions:
                            _trap_ions[key] = []
                        if len(_trap_ions[key]) < wiseArch.k:
                            _trap_ions[key].append(ion)
                        else:
                            _unplaced.append(ion)
                    else:
                        _unplaced.append(ion)

                # Place remaining ions (bridge ancillas etc.) at edge traps
                for ion in _unplaced:
                    _placed = False
                    for r in [0, block_rows - 1]:
                        if _placed:
                            break
                        for tc in range(block_cols):
                            key = (r, tc)
                            if key not in _trap_ions:
                                _trap_ions[key] = []
                            if len(_trap_ions[key]) < wiseArch.k:
                                _trap_ions[key].append(ion)
                                _placed = True
                                break
                    if not _placed:
                        for r in range(block_rows):
                            if _placed:
                                break
                            for tc in range(block_cols):
                                key = (r, tc)
                                if key not in _trap_ions:
                                    _trap_ions[key] = []
                                if len(_trap_ions[key]) < wiseArch.k:
                                    _trap_ions[key].append(ion)
                                    _placed = True
                                    break

                # Convert to trap_for_grid format: (ions, pairs) tuple
                for (local_r, local_c), ions_list in _trap_ions.items():
                    global_c = local_c + c0
                    global_r = local_r + r0
                    trap_for_grid[(2 * global_c, global_r)] = (ions_list, [])

                # Update sub-grid layout array for downstream use
                import numpy as np
                layout = np.zeros((block_rows, block_cols * wiseArch.k), dtype=int)
                ion_indices = []
                for (local_r, local_c), ions_list in _trap_ions.items():
                    for slot_i, ion in enumerate(ions_list):
                        col_idx = local_c * wiseArch.k + slot_i
                        if local_r < block_rows and col_idx < block_cols * wiseArch.k:
                            layout[local_r, col_idx] = ion.idx
                            ion_indices.append(ion.idx)
                sg.initial_layout = layout
                sg.ion_indices = ion_indices
                continue

            # -----------------------------------------------------------
            # Fallback: stochastic regularPartition + hillClimb
            # -----------------------------------------------------------
            max_clusters = block_rows * block_cols
            clusters = regularPartition(
                m_ions, d_ions, wiseArch.k,
                isWISEArch=self.compact_clustering,
                maxClusters=max_clusters,
            )

            # Block-local grid positions for hillClimb
            block_grid_pos = [
                (c, r)
                for r in range(block_rows)
                for c in range(block_cols)
            ]
            grid_positions = hillClimbOnArrangeClusters(
                clusters, allGridPos=block_grid_pos
            )

            # Map block-local positions to global positions
            for cluster_idx, (local_c, local_r) in enumerate(grid_positions):
                global_c = local_c + c0
                global_r = local_r + r0
                trap_for_grid[(2 * global_c, global_r)] = clusters[cluster_idx]

        # --- Build architecture topology (identical to build_topology) ---
        rows = wiseArch.n
        cols = wiseArch.m
        self._originalArrangement = {}

        traps_dict: Dict[Tuple[int, int], Trap] = {}
        for row in range(rows):
            for col in range(cols):
                if (2 * col, row) in trap_for_grid:
                    ions = trap_for_grid[(2 * col, row)][0]
                else:
                    ions = []
                if self.add_spectators:
                    maxIdx = max(ion_mapping.keys()) if ion_mapping else 0
                    nplaceholds = wiseArch.k - len(ions)
                    for i in range(nplaceholds):
                        ion = SpectatorIon("#bbbbbb", "P")
                        idx = maxIdx + 1 + i
                        ion.set(idx, *ion.pos)
                        ion_mapping[idx] = ion, ion.pos
                        ions.append(ion)
                traps_dict[(2 * col, row)] = self.addManipulationTrap(
                    *self._gridToCoordinate((2 * col, row), wiseArch.k),
                    ions,
                    isHorizontal=True,
                    capacity=wiseArch.k,
                )
                self._originalArrangement[traps_dict[(2 * col, row)]] = ions

        if rows == 1:
            for (col, r), trap_node in traps_dict.items():
                if (col + 2, r) in traps_dict:
                    self.addEdge(trap_node, traps_dict[(col + 2, r)])
        else:
            junctions_dict: Dict[Tuple[int, int], Junction] = {}
            for (col, row), trap_node in traps_dict.items():
                if (col, row + 1) in traps_dict:
                    if (col + 1, row) not in junctions_dict:
                        junctionTop = self.addJunction(
                            *self._gridToCoordinate(
                                (col + 1, row), wiseArch.k
                            ),
                        )
                        junctions_dict[(col + 1, row)] = junctionTop
                    if (col + 1, row + 1) not in junctions_dict:
                        junctionBottom = self.addJunction(
                            *self._gridToCoordinate(
                                (col + 1, row + 1), wiseArch.k
                            ),
                        )
                        junctions_dict[(col + 1, row + 1)] = junctionBottom

            for (col, row), junctionTop in junctions_dict.items():
                if (col, row + 1) in junctions_dict:
                    self.addEdge(
                        junctions_dict[(col, row + 1)], junctionTop
                    )

            for row in range(rows):
                for col in range(cols):
                    if (2 * col + 1, row) in junctions_dict and (
                        2 * col, row,
                    ) in traps_dict:
                        self.addEdge(
                            junctions_dict[(2 * col + 1, row)],
                            traps_dict[(2 * col, row)],
                        )
                    if (2 * col + 2, row) in traps_dict and (
                        2 * col + 1, row,
                    ) in junctions_dict:
                        self.addEdge(
                            traps_dict[(2 * col + 2, row)],
                            junctions_dict[(2 * col + 1, row)],
                        )

        if any(i.parent is None for i in self.ions.values()):
            raise ValueError(
                f"Ions not in traps — per-block build_topology_per_block "
                f"with k={wiseArch.k}, blocks={list(block_sub_grids.keys())}"
            )


# ============================================================================
# NetworkedGridArchitecture
# ============================================================================


class NetworkedGridArchitecture(QCCDArch):
    """QCCD topology with a fully-connected grid via junction chain.

    A single column of traps, each connected via a junction chain to
    every other trap.

    Parameters
    ----------
    trap_capacity : int
        Number of ion slots per trap.
    num_traps : int
        Number of traps in the column.
    """

    def __init__(self, trap_capacity: int = 2, num_traps: int = 1):
        super().__init__()
        self.trap_capacity = trap_capacity
        self.num_traps = num_traps

    def build_topology(
        self,
        measurement_ions: Sequence[Ion],
        data_ions: Sequence[Ion],
        ion_mapping: Dict[int, Tuple[Ion, Tuple[int, int]]],
    ) -> None:
        """Build the networked grid topology.

        Mirrors the logic previously in
        ``QCCDCircuit.processCircuitNetworkedGrid``.

        Parameters
        ----------
        measurement_ions : Sequence[Ion]
            Measurement (ancilla) ions.
        data_ions : Sequence[Ion]
            Data ions.
        ion_mapping : Dict[int, Tuple[Ion, Tuple[int, int]]]
            ``{stim_idx: (Ion, coords)}`` mapping.
        """
        trapCapacity = self.trap_capacity
        traps = self.num_traps
        if (trapCapacity - 1) * traps < len(ion_mapping):
            raise ValueError("processCircuit: not enough traps")

        clusters = regularPartition(
            measurement_ions, data_ions, trapCapacity
        )

        allGridPos = []
        for r in range(traps):
            allGridPos.append((0, r))

        gridPositions = arrangeClusters(clusters, allGridPos=allGridPos)

        trap_for_grid = {
            row: clusters[trapIdx]
            for trapIdx, (_, row) in enumerate(gridPositions)
        }
        self._originalArrangement = {}

        traps_dict: Dict[int, Trap] = {}
        for row in range(traps):
            if row in trap_for_grid:
                ions = trap_for_grid[row][0]
            else:
                ions = []
            traps_dict[row] = self.addManipulationTrap(
                *self._gridToCoordinate((0, row), trapCapacity),
                ions,
                isHorizontal=True,
                capacity=trapCapacity,
            )
            self._originalArrangement[traps_dict[row]] = ions

        switch_cost = 1
        junctions_dict: Dict[Tuple[int, int], Junction] = {}

        for row, trap_node in traps_dict.items():
            for i in range(switch_cost):
                junction2 = self.addJunction(
                    *self._gridToCoordinate((i + 1, row), trapCapacity),
                )
                junctions_dict[(i + 1, row)] = junction2
                if i == 0:
                    self.addEdge(trap_node, junction2)
                else:
                    self.addEdge(junctions_dict[(i, row)], junction2)

        for row, trap_node in traps_dict.items():
            junction2 = junctions_dict[(switch_cost, row)]
            for row2 in range(traps):
                if row == row2:
                    continue
                junction1 = junctions_dict[(switch_cost, row2)]
                self.addEdge(junction1, junction2)
