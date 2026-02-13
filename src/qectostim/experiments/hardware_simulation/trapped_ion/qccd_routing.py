# src/qectostim/experiments/hardware_simulation/trapped_ion/qccd_routing.py
"""
Junction-based QCCD ion routing.

Ports the old ``ionRouting()`` algorithm from ``qccd_ion_routing.py`` into
the new architecture.  This is the **non-SAT** routing strategy used for
general QCCD architectures (as opposed to the WISE SAT-based router).

Algorithm overview
------------------
1. Greedily execute any gate whose ions already share a trap.
2. For remaining gates, find shortest path through the junction graph
   (via ``networkx.all_shortest_paths``) to shuttle the ancilla ion
   to the data ion's trap.
3. If the destination trap is full, plan a "go-back" return trip to
   the nearest trap with capacity along the reverse path.
4. Generate barriers that synchronise concurrent movements.

References
----------
* Original implementation: ``old/src/compiler/qccd_ion_routing.py``
* Transport operations: :mod:`transport`
* Architecture graph: :class:`architecture.QCCDGraph`
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import (
    Dict,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
)

import networkx as nx

from qectostim.experiments.hardware_simulation.trapped_ion.architecture import (
    Ion,
    QCCDGraph,
    QCCDNode,
    ManipulationTrap,
    StorageTrap,
    Junction,
    Crossing,
)
from qectostim.experiments.hardware_simulation.trapped_ion.transport import (
    TransportOp,
    build_hop_operations,
    total_transport_time,
    total_transport_heating,
)

logger = logging.getLogger(__name__)


# ============================================================================
# Data structures
# ============================================================================

@dataclass
class RoutingBarrier:
    """Marks a synchronisation point in the operation sequence.

    All operations before ``op_index`` must complete before the next
    batch begins.
    """
    op_index: int


@dataclass
class GateRequest:
    """A pending two-qubit gate that may require ion routing."""
    ancilla_ion: Ion
    data_ion: Ion
    priority: int       # original gate order (lower = higher priority)
    gate_id: int        # index into the external gate list


@dataclass
class QCCDRoutingResult:
    """Output of the QCCD junction-based router.

    Attributes
    ----------
    transport_ops : list[TransportOp]
        Flat list of transport operations in execution order.
    barriers : list[int]
        Indices into ``transport_ops`` marking synchronisation barriers.
    total_time_s : float
        Estimated sequential execution time (sum of op times).
    total_heating : float
        Total motional quanta deposited across all transport ops.
    gate_execution_order : list[int]
        Order in which the original gate_ids were executed.
    """
    transport_ops: List[TransportOp] = field(default_factory=list)
    barriers: List[int] = field(default_factory=list)
    total_time_s: float = 0.0
    total_heating: float = 0.0
    gate_execution_order: List[int] = field(default_factory=list)


# ============================================================================
# Main routing function
# ============================================================================

def route_ions_junction(
    graph: QCCDGraph,
    gate_requests: Sequence[GateRequest],
    trap_capacity: int,
    max_rounds: int = 200,
    timeout_s: float = 30.0,
) -> QCCDRoutingResult:
    """Route ions through a QCCD junction network.

    This is a faithful port of the old ``ionRouting()`` function:

    1.  Greedily execute gates whose ions already share a trap.
    2.  For remaining gates, compute all shortest paths from ancilla
        to data-ion trap via ``nx.all_shortest_paths``.
    3.  Pick the first conflict-free path (no shared crossings, no
        overfull nodes).
    4.  If destination is at capacity, plan a go-back trip.
    5.  Step ions forward hop-by-hop, generating transport ops.
    6.  Record barriers between rounds.

    Parameters
    ----------
    graph : QCCDGraph
        The QCCD architecture graph (must have ``build_networkx_graph()``
        called and up-to-date).
    gate_requests : sequence of GateRequest
        Pending two-qubit gates, each with ancilla + data ion refs.
    trap_capacity : int
        Maximum ions per trap (used for capacity checks).

    Returns
    -------
    QCCDRoutingResult
        Transport operations, barriers, and timing information.
    """
    import time as _time

    result = QCCDRoutingResult()
    remaining = list(gate_requests)
    move_candidates: Dict[int, GateRequest] = {}  # ancilla_idx → request

    _t0 = _time.monotonic()
    _round = 0

    while remaining:
        _round += 1

        # --- Hard round limit ---
        if _round > max_rounds:
            logger.warning(
                "QCCD routing: exceeded %d rounds, %d gates unrouted — aborting",
                max_rounds, len(remaining),
            )
            break

        # --- Wall-clock timeout ---
        if _time.monotonic() - _t0 > timeout_s:
            logger.warning(
                "QCCD routing: %.1fs timeout after %d rounds, "
                "%d gates unrouted — aborting",
                timeout_s, _round, len(remaining),
            )
            break
        # ------------------------------------------------------------------
        # Phase 1: Execute all gates that need no routing
        # ------------------------------------------------------------------
        changed = True
        while changed:
            changed = False
            to_remove: List[GateRequest] = []
            ions_busy: Set[int] = set()

            for req in remaining:
                a_idx = req.ancilla_ion.idx
                d_idx = req.data_ion.idx
                # Skip if either ion is already in use this sub-round
                if a_idx in ions_busy or d_idx in ions_busy:
                    ions_busy.add(a_idx)
                    ions_busy.add(d_idx)
                    continue
                # Check co-location
                a_parent = req.ancilla_ion.parent
                d_parent = req.data_ion.parent
                if (
                    a_parent is not None
                    and d_parent is not None
                    and a_parent is d_parent
                    and isinstance(a_parent, ManipulationTrap)
                ):
                    # Gate can execute in-place
                    to_remove.append(req)
                    result.gate_execution_order.append(req.gate_id)
                ions_busy.add(a_idx)
                ions_busy.add(d_idx)

            for req in to_remove:
                remaining.remove(req)
                changed = True

        if not remaining:
            break

        # ------------------------------------------------------------------
        # Phase 2: Identify gates that need routing
        # ------------------------------------------------------------------
        for req in remaining:
            a_idx = req.ancilla_ion.idx
            if a_idx in move_candidates:
                continue
            a_parent = req.ancilla_ion.parent
            d_parent = req.data_ion.parent
            if a_parent is not None and a_parent is d_parent:
                continue  # already co-located
            move_candidates[a_idx] = req

        # Sort candidates by original priority
        sorted_moves = sorted(
            move_candidates.items(), key=lambda kv: kv[1].priority
        )

        # ------------------------------------------------------------------
        # Phase 3: Plan movements (pick conflict-free shortest paths)
        # ------------------------------------------------------------------
        crossings_used: Set[int] = set()
        nodes_full: Set[int] = set()
        # ancilla_idx → (request, path_nodes, dest_trap)
        movements: Dict[int, Tuple[GateRequest, List[QCCDNode], QCCDNode]] = {}

        nxg = graph.graph  # the underlying nx.DiGraph

        for a_idx, req in sorted_moves:
            d_parent = req.data_ion.parent
            if d_parent is None or not isinstance(d_parent, (ManipulationTrap, StorageTrap)):
                continue

            a_parent = req.ancilla_ion.parent
            if a_parent is None:
                continue

            src_id = a_parent.idx
            dst_id = d_parent.idx

            if src_id == dst_id:
                continue  # already co-located

            try:
                all_paths = list(nx.all_shortest_paths(nxg, src_id, dst_id, weight='weight'))
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                continue

            chosen_nodes: List[QCCDNode] = []
            chosen_crossing_ids: List[int] = []

            for path in all_paths:
                # Collect crossings along this path
                path_crossings: List[int] = []
                for n1, n2 in zip(path[:-1], path[1:]):
                    edge_key = (n1, n2)
                    if edge_key in graph._crossing_edges:
                        path_crossings.append(graph._crossing_edges[edge_key].idx)

                # Collect nodes along this path
                path_nodes: List[QCCDNode] = []
                for nid in path:
                    if nid in graph.nodes:
                        nd = graph.nodes[nid]
                        if nd not in path_nodes:
                            path_nodes.append(nd)

                # Check for capacity conflicts (skip source)
                full_in_path: List[int] = []
                for nd in path_nodes[1:]:
                    if isinstance(nd, Junction) and nd.num_ions >= 1:
                        full_in_path.append(nd.idx)
                    elif nd.num_ions >= trap_capacity:
                        full_in_path.append(nd.idx)

                # Check for crossing / node conflicts
                if (
                    crossings_used.isdisjoint(path_crossings)
                    and nodes_full.isdisjoint(full_in_path)
                ):
                    chosen_nodes = path_nodes
                    chosen_crossing_ids = path_crossings
                    break

            if not chosen_nodes:
                continue  # can't route this round

            # Reserve
            move_candidates.pop(a_idx)
            movements[a_idx] = (req, chosen_nodes, d_parent)
            crossings_used.update(chosen_crossing_ids)
            for nd in chosen_nodes[1:]:
                if isinstance(nd, Junction) and nd.num_ions + 1 >= 1:
                    nodes_full.add(nd.idx)
                elif nd.num_ions + 1 >= trap_capacity:
                    nodes_full.add(nd.idx)

        # ------------------------------------------------------------------
        # Phase 4: Determine go-back targets for full destination traps
        # ------------------------------------------------------------------
        # Stale-progress guard: if Phase 3 found NOTHING to route, we are
        # stuck — break to avoid spinning forever.
        if not movements:
            logger.warning(
                "QCCD routing: no valid routes for %d remaining gates "
                "after %d rounds — aborting.  Unrouted gate IDs: %s",
                len(remaining), _round,
                [r.gate_id for r in remaining[:10]],
            )
            break

        # ancilla_idx → (request, path_nodes, go_back_trap_or_None)
        to_forward: Dict[int, Tuple[GateRequest, List[QCCDNode], Optional[QCCDNode]]] = {}

        for a_idx, (req, path_nodes, dest_trap) in movements.items():
            if dest_trap.num_ions >= trap_capacity:
                # Find a trap with room along the reverse path
                go_back = path_nodes[0]  # fallback: source
                for nd in reversed(path_nodes[:-1]):
                    if isinstance(nd, (ManipulationTrap, StorageTrap)) and nd.num_ions < trap_capacity:
                        go_back = nd
                        break
                to_forward[a_idx] = (req, path_nodes, go_back)
            else:
                to_forward[a_idx] = (req, path_nodes, None)

        # ------------------------------------------------------------------
        # Phase 5: Step ions forward, generating transport ops
        # ------------------------------------------------------------------
        gone_back: Set[int] = set()  # ancilla indices that have started returning

        while to_forward:
            ions_busy_fwd: Set[int] = set()
            done_this_round: List[int] = []

            sorted_fwd = sorted(
                to_forward.items(),
                key=lambda kv: kv[1][0].priority,
            )

            for a_idx, (req, path_nodes, go_back_trap) in sorted_fwd:
                if a_idx in ions_busy_fwd:
                    continue

                ion = req.ancilla_ion
                current_parent = ion.parent
                if current_parent is None:
                    continue

                dest = path_nodes[-1]
                going_back = a_idx in gone_back

                # Check if arrived at final destination
                if not going_back and current_parent is dest:
                    # Gate can execute
                    result.gate_execution_order.append(req.gate_id)
                    remaining = [r for r in remaining if r.gate_id != req.gate_id]
                    ions_busy_fwd.add(a_idx)
                    if go_back_trap is not None:
                        gone_back.add(a_idx)
                    else:
                        done_this_round.append(a_idx)
                    continue

                # Check if returned to go-back trap
                if going_back and go_back_trap is not None and current_parent is go_back_trap:
                    done_this_round.append(a_idx)
                    continue

                # Determine next node in the path
                current_idx = -1
                for i, nd in enumerate(path_nodes):
                    if nd is current_parent:
                        current_idx = i
                        break

                if current_idx < 0:
                    # Ion is not on the expected path — skip
                    continue

                if going_back:
                    if current_idx <= 0:
                        done_this_round.append(a_idx)
                        continue
                    next_node = path_nodes[current_idx - 1]
                else:
                    if current_idx >= len(path_nodes) - 1:
                        continue
                    next_node = path_nodes[current_idx + 1]

                # Generate hop operations between current and next node
                src_node = path_nodes[current_idx]
                # Find the crossing between src_node and next_node
                crossing_key = (src_node.idx, next_node.idx)
                alt_key = (next_node.idx, src_node.idx)
                crossing = graph._crossing_edges.get(crossing_key) or graph._crossing_edges.get(alt_key)

                if crossing is None:
                    logger.warning(
                        f"No crossing between {src_node.idx} and {next_node.idx}"
                    )
                    continue

                hop_ops = build_hop_operations(
                    source_idx=src_node.idx,
                    target_idx=next_node.idx,
                    crossing_idx=crossing.idx,
                    ion_idx=ion.idx,
                    source_is_junction=isinstance(src_node, Junction),
                    target_is_junction=isinstance(next_node, Junction),
                    needs_rotation=(
                        src_node.num_ions == 1
                        and not isinstance(src_node, Junction)
                    ),
                )

                result.transport_ops.extend(hop_ops)
                ions_busy_fwd.add(a_idx)

                # Simulate the state change: move ion from src to dest
                if ion in src_node.ions:
                    src_node.remove_ion(ion)
                next_node.add_ion(ion, force=True)  # allow temporary over-capacity

                # Apply heating to affected ions
                for op in hop_ops:
                    heating = op.heating_quanta
                    ion.add_motional_energy(heating)
                    src_node.add_motional_energy(heating)

            # Remove completed movements
            for a_idx in done_this_round:
                to_forward.pop(a_idx, None)
                gone_back.discard(a_idx)

            # Safety: if nothing moved, break to avoid infinite loop
            if not done_this_round and not ions_busy_fwd:
                logger.warning("QCCD routing: deadlock detected, breaking")
                break

        # Record barrier
        result.barriers.append(len(result.transport_ops))

        # Refresh graph after this round of movements
        graph.refresh_graph()

    # Compute totals
    result.total_time_s = total_transport_time(result.transport_ops)
    result.total_heating = total_transport_heating(result.transport_ops)

    _elapsed = _time.monotonic() - _t0
    logger.info(
        "QCCD routing: %d/%d gates routed in %d rounds (%.3fs), "
        "%d transport ops",
        len(result.gate_execution_order), len(gate_requests),
        _round, _elapsed, len(result.transport_ops),
    )

    return result
