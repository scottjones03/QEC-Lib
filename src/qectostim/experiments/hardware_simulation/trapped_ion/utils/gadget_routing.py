"""
Phase-aware routing for fault-tolerant gadget experiments.

This module implements the multi-block routing pipeline described in
GADGET_COMPILATION_SPEC.md, including:

- MS pair derivation from QECMetadata (Step 2)
- Sub-grid partitioning and per-block mapping (Step 3)
- Ion-return BT constraints (Step 4)
- Cached round replay (Step 5)
- Block schedule merging (Step 6)
- Gadget phase routing (Step 7)

Key invariant: different blocks occupy **disjoint** sub-grid regions,
enabling independent per-block SAT solving (Level 1 spatial slicing).
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field, replace as _dc_replace
from typing import (
    Collection,
    Dict,
    List,
    Optional,
    Set,
    Tuple,
    Any,
    TYPE_CHECKING,
)

import numpy as np

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from qectostim.experiments.hardware_simulation.core.pipeline import (
        QECMetadata,
    )
    from qectostim.gadgets.base import Gadget
    from qectostim.gadgets.layout import QubitAllocation


# =====================================================================
# Dataclasses
# =====================================================================

@dataclass
class BlockSubGrid:
    """Sub-grid allocation for one logical block.

    Attributes
    ----------
    block_name : str
        Name matching ``BlockInfo.block_name`` / ``BlockAllocation.block_name``.
    grid_region : Tuple[int, int, int, int]
        ``(r0, c0, r1, c1)`` on the full grid (row-half-open, col-half-open).
    n_rows : int
        Sub-grid row count (``r1 - r0``).
    n_cols : int
        Sub-grid column count (``c1 - c0``).
    ion_indices : List[int]
        Ion indices assigned to this sub-grid.
    qubit_to_ion : Dict[int, int]
        Stim qubit index → ion index mapping for this block.
    initial_layout : Optional[np.ndarray]
        ``n_rows × n_cols`` ion arrangement array (populated by
        ``map_qubits_per_block``).
    """
    block_name: str = ""
    grid_region: Tuple[int, int, int, int] = (0, 0, 0, 0)
    n_rows: int = 0
    n_cols: int = 0
    ion_indices: List[int] = field(default_factory=list)
    qubit_to_ion: Dict[int, int] = field(default_factory=dict)
    initial_layout: Optional[np.ndarray] = None
    coord_to_trap_pos: Optional[Dict[Tuple[int, int], Tuple[int, int, int]]] = None


@dataclass
class PhaseRoutingPlan:
    """Plan for routing one temporal phase of a gadget experiment.

    Attributes
    ----------
    phase_type : str
        ``"ec"``, ``"gadget"``, or ``"transition"``.
    phase_index : int
        Index into ``QECMetadata.phases``.
    interacting_blocks : List[str]
        Blocks that interact (merged onto shared grid).
    idle_blocks : List[str]
        Blocks EXCLUDED from this phase's SAT grid.
    all_blocks : List[str]
        ``interacting + idle`` (for bookkeeping).
    ms_pairs_per_round : List[List[Tuple[int, int]]]
        Ion-index pairs per MS round.
    num_rounds : int
        How many times this phase repeats.
    is_cached : bool
        ``True`` if solved once and replayed.
    grid_region : Tuple[int, int, int, int]
        ``(r0, c0, r1, c1)`` sub-grid for this phase.
    round_signature : Optional[Tuple]
        Canonical key for cache lookup.
    identical_to_phase : Optional[int]
        Phase index with the same routing solution.
    """
    phase_type: str = ""
    phase_index: int = 0
    interacting_blocks: List[str] = field(default_factory=list)
    idle_blocks: List[str] = field(default_factory=list)
    all_blocks: List[str] = field(default_factory=list)
    ms_pairs_per_round: List[List[Tuple[int, int]]] = field(default_factory=list)
    num_rounds: int = 0
    is_cached: bool = False
    grid_region: Tuple[int, int, int, int] = (0, 0, 0, 0)
    round_signature: Optional[Tuple] = None
    identical_to_phase: Optional[int] = None
    round_labels: List[str] = field(default_factory=list)
    """Per-round label: ``"bridge"`` for merge CX rounds,
    ``"ec"`` for interleaved EC stabiliser rounds, ``"ms"`` for
    generic MS rounds.  Empty list when labels are not computed."""


# =====================================================================
# Step 2: MS Pair Derivation from Metadata
# =====================================================================

def derive_ms_pairs_from_metadata(
    qec_meta: "QECMetadata",
    qubit_to_ion: Dict[int, int],
    block_name: Optional[str] = None,
) -> List[List[Tuple[int, int]]]:
    """Convert CNOT schedule from metadata to ion-index MS pairs.

    Returns one list-of-pairs per parallel MS round (e.g. 4 rounds for
    rotated surface code: N, E, S, W CNOT layers → 4 MS rounds; when
    X+Z are combined there are 8 layers).

    Parameters
    ----------
    qec_meta : QECMetadata
        Metadata with stabilizer CNOT schedules.
    qubit_to_ion : Dict[int, int]
        Stim qubit index → ion index mapping.
    block_name : str, optional
        If specified, uses ``per_block_stabilizers[block_name]`` instead
        of the global X/Z stabilizer schedules, and filters to only
        qubits belonging to that block.

    Returns
    -------
    List[List[Tuple[int, int]]]
        One inner list per parallel MS round.  Each pair is
        ``(ion_ctrl, ion_tgt)`` — typically ``(ancilla_ion, data_ion)``.
    """
    ms_rounds: List[List[Tuple[int, int]]] = []

    if block_name and block_name in qec_meta.per_block_stabilizers:
        # Use per-block stabilizer info (combined X+Z schedule)
        block_stab = qec_meta.per_block_stabilizers[block_name]
        cnot_sched = block_stab.cnot_schedule or []

        # Build set of qubits belonging to this block for filtering
        block_qubits: Set[int] = set()
        for ba in qec_meta.block_allocations:
            if ba.block_name == block_name:
                block_qubits.update(ba.data_qubits)
                block_qubits.update(ba.x_ancilla_qubits)
                block_qubits.update(ba.z_ancilla_qubits)
                break

        for layer in cnot_sched:
            pairs: List[Tuple[int, int]] = []
            for ctrl, tgt in layer:
                # Filter: both qubits must belong to this block
                if block_qubits and (
                    ctrl not in block_qubits or tgt not in block_qubits
                ):
                    continue
                # Both qubits must have ion mappings
                if ctrl not in qubit_to_ion or tgt not in qubit_to_ion:
                    continue
                pairs.append((qubit_to_ion[ctrl], qubit_to_ion[tgt]))
            if pairs:
                ms_rounds.append(pairs)
    else:
        # Use global X/Z stabilizer schedules — interleave by phase
        x_sched = qec_meta.x_stabilizers.cnot_schedule or []
        z_sched = qec_meta.z_stabilizers.cnot_schedule or []

        # If block_name is given but not in per_block_stabilizers,
        # filter the global schedules to this block's qubits
        block_qubits_global: Optional[Set[int]] = None
        if block_name:
            block_qubits_global = set()
            for ba in qec_meta.block_allocations:
                if ba.block_name == block_name:
                    block_qubits_global.update(ba.data_qubits)
                    block_qubits_global.update(ba.x_ancilla_qubits)
                    block_qubits_global.update(ba.z_ancilla_qubits)
                    break

        # Combine X+Z layers respecting the scheduling mode.
        # Interleaved (surface codes): X[i]+Z[i] merged per phase.
        # Sequential (non-geometric / different-length schedules):
        #   all X layers first, then all Z layers.
        _is_il = getattr(qec_meta, 'is_interleaved', None)
        if _is_il is None:
            # Legacy fallback: interleave when schedule lengths match
            _is_il = (len(x_sched) > 0 and len(z_sched) > 0
                      and len(x_sched) == len(z_sched))

        if _is_il:
            # Interleave X+Z layers by phase index
            n_layers = max(len(x_sched), len(z_sched))
            combined_iter: List[List[Tuple[int, int]]] = []
            for li in range(n_layers):
                merged_layer: List[Tuple[int, int]] = []
                if li < len(x_sched):
                    merged_layer.extend(x_sched[li])
                if li < len(z_sched):
                    merged_layer.extend(z_sched[li])
                combined_iter.append(merged_layer)
        else:
            # Sequential: all X layers then all Z layers
            combined_iter = [list(layer) for layer in x_sched]
            combined_iter += [list(layer) for layer in z_sched]

        for merged_layer in combined_iter:
            pairs = []
            for ctrl, tgt in merged_layer:
                if block_qubits_global and (
                    ctrl not in block_qubits_global
                    or tgt not in block_qubits_global
                ):
                    continue
                if ctrl not in qubit_to_ion or tgt not in qubit_to_ion:
                    continue
                pairs.append((qubit_to_ion[ctrl], qubit_to_ion[tgt]))
            if pairs:
                ms_rounds.append(pairs)

    return ms_rounds


def derive_gadget_ms_pairs(
    gadget: "Gadget",
    phase_index: int,
    qubit_allocation: "QubitAllocation",
    interacting_blocks: List[str],
    qubit_to_ion: Dict[int, int],
) -> List[List[Tuple[int, int]]]:
    """Extract MS pairs for a specific gadget phase.

    Works for any 2-block interaction: TransversalCNOT, KnillEC phases,
    CSSSurgery merge/split phases, teleportation gadgets.

    For transversal gadgets, the pairs are data_block_a[i] ↔ data_block_b[i]
    (one parallel MS round).  For gadgets with more complex interactions,
    the pairs come from ``gadget.get_phase_pairs(phase_index)``.

    Parameters
    ----------
    gadget : Gadget
        The gadget object.
    phase_index : int
        Which gadget phase (0-indexed).
    qubit_allocation : QubitAllocation
        Full allocation with all blocks.
    interacting_blocks : List[str]
        The 2 blocks that interact in this phase.
    qubit_to_ion : Dict[int, int]
        Global qubit → ion mapping.

    Returns
    -------
    List[List[Tuple[int, int]]]
        One inner list per parallel MS round.  Pairs are
        ``(ion_ctrl, ion_tgt)``.
    """
    # Try the gadget's get_phase_pairs method first
    # M4 fix: pass qubit_allocation as required; handle List[List[Tuple]] return
    if hasattr(gadget, "get_phase_pairs"):
        raw_rounds = gadget.get_phase_pairs(phase_index, qubit_allocation)
        ms_rounds: List[List[Tuple[int, int]]] = []
        for round_pairs in raw_rounds:
            ion_round: List[Tuple[int, int]] = []
            for ctrl_q, tgt_q in round_pairs:
                if ctrl_q in qubit_to_ion and tgt_q in qubit_to_ion:
                    ion_round.append(
                        (qubit_to_ion[ctrl_q], qubit_to_ion[tgt_q])
                    )
            if ion_round:
                ms_rounds.append(ion_round)
        return ms_rounds

    # Fallback: for transversal gadgets, derive from block data qubit pairing
    # TransversalCNOT: pairs block_a.data[i] ↔ block_b.data[i]
    if len(interacting_blocks) < 2:
        # Single-block or empty phase — no 2Q interactions in fallback mode
        return []
    if len(interacting_blocks) > 2:
        logger.warning(
            "derive_gadget_ms_pairs: transversal fallback only supports 2 "
            "blocks, got %d: %s — returning empty pairs",
            len(interacting_blocks), interacting_blocks,
        )
        return []

    block_a = qubit_allocation.blocks[interacting_blocks[0]]
    block_b = qubit_allocation.blocks[interacting_blocks[1]]

    # Pair data qubits by position: a.data[i] ↔ b.data[i]
    a_data = list(block_a.data_range)
    b_data = list(block_b.data_range)

    if len(a_data) != len(b_data):
        raise ValueError(
            f"Transversal fallback: blocks have different data counts: "
            f"{interacting_blocks[0]}={len(a_data)}, "
            f"{interacting_blocks[1]}={len(b_data)}"
        )

    round_pairs = []
    for qa, qb in zip(a_data, b_data):
        if qa in qubit_to_ion and qb in qubit_to_ion:
            round_pairs.append((qubit_to_ion[qa], qubit_to_ion[qb]))

    return [round_pairs] if round_pairs else []


# =====================================================================
# Step 3: Sub-Grid Partitioning (§3.1.1)
# =====================================================================

def _block_grid_dims(n_qubits: int, k: int) -> Tuple[int, int]:
    """Return ``(n_rows, n_cols)`` for a single block's sub-grid."""
    n_rows = int(math.ceil(math.sqrt(n_qubits)))
    n_cols = max(int(math.ceil(n_qubits / (n_rows * k))), 1)
    return n_rows, n_cols


def allocate_block_regions(
    block_names: List[str],
    qubits_per_block: Dict[str, int],
    k: int,
    gap_cols: int = 0,
    block_offsets: Optional[Dict[str, Tuple[float, ...]]] = None,
) -> Dict[str, Tuple[int, int, int, int]]:
    """Allocate non-overlapping ``(r0, c0, r1, c1)`` regions per block.

    When *block_offsets* is provided (from ``QubitAllocation.blocks``),
    the gadget's spatial geometry is preserved on the WISE grid:
    blocks are placed at grid positions that reflect their relative
    ``(x, y)`` offsets.  This is critical for Level-1 spatial slicing —
    e.g. CSS Surgery's vertical column layout becomes a vertical stack
    of sub-grids, and any future L-shaped layout would produce an
    L-shaped grid allocation.

    When *block_offsets* is ``None`` the legacy side-by-side column
    layout is used (blocks placed left-to-right at row 0).

    Parameters
    ----------
    block_names : List[str]
        Ordered list of block names.
    qubits_per_block : Dict[str, int]
        Total qubit (ion) count per block.
    k : int
        Trap capacity.
    gap_cols : int
        Gap columns between adjacent blocks.
    block_offsets : Dict[str, Tuple[float, ...]], optional
        Gadget-layout offsets per block (``(x, y)``).  When present the
        spatial geometry is preserved.

    Returns
    -------
    Dict[str, Tuple[int, int, int, int]]
        ``{block_name: (r0, c0, r1, c1)}``.
        Row range: ``[r0, r1)``.  Column range: ``[c0, c1)``.

    Invariant
    ---------
    For all block pairs ``(A, B)``:
        ``region_A ∩ region_B == ∅``  (no shared row × column cells).
    """
    # ------------------------------------------------------------------
    # Compute per-block sub-grid dimensions
    # ------------------------------------------------------------------
    dims: Dict[str, Tuple[int, int]] = {}
    for name in block_names:
        dims[name] = _block_grid_dims(qubits_per_block[name], k)

    # ------------------------------------------------------------------
    # Layout-preserving placement (uses gadget block offsets)
    # ------------------------------------------------------------------
    if block_offsets is not None and len(block_offsets) >= 2:
        return _allocate_regions_from_offsets(
            block_names, dims, block_offsets, gap_cols,
        )

    # ------------------------------------------------------------------
    # Legacy fallback: side-by-side along column axis
    # ------------------------------------------------------------------
    regions: Dict[str, Tuple[int, int, int, int]] = {}
    col_cursor = 0
    for name in block_names:
        n_rows, n_cols = dims[name]
        regions[name] = (0, col_cursor, n_rows, col_cursor + n_cols)
        col_cursor += n_cols + gap_cols
    return regions


def _allocate_regions_from_offsets(
    block_names: List[str],
    dims: Dict[str, Tuple[int, int]],
    block_offsets: Dict[str, Tuple[float, ...]],
    gap: int,
) -> Dict[str, Tuple[int, int, int, int]]:
    """Place blocks on the grid preserving their spatial layout.

    Strategy
    --------
    1. Normalise block offsets so the top-left-most block is at the
       origin.
    2. Quantise each block's continuous ``(x, y)`` centre to a grid
       position (row, col).  X maps to columns, Y maps to rows.  The
       gadget coordinate system has **Y increasing upward**, while the
       grid has **row 0 at the top**, so Y is *negated* when mapping to
       rows.
    3. After initial placement, resolve any overlaps by nudging blocks
       apart (greedy, respects adjacency direction).
    4. Guarantee at least *gap* empty columns/rows between every pair
       of regions.
    """
    # Step 1: gather offsets; fall back to (0, 0) for missing blocks
    raw: Dict[str, Tuple[float, float]] = {}
    for name in block_names:
        off = block_offsets.get(name, (0.0, 0.0))
        raw[name] = (float(off[0]), float(off[1]) if len(off) > 1 else 0.0)

    # Step 2: compute relative positions.
    #   x → col direction (left = small col)
    #   y → row direction (y up ↔ row 0 at top, so negate)
    # Normalise so the minimum (col, row) centre is at (0, 0).
    #
    # We quantise by sorting blocks along each axis and placing them
    # at sequential grid positions based on their sub-grid size.
    # This preserves *order and adjacency* without requiring exact
    # continuous-to-discrete coordinate conversion — important because
    # the qubit coordinate space uses half-integers while the grid
    # uses integer trap positions.

    # Determine distinct "slots" per axis via clustering
    x_vals = {name: raw[name][0] for name in block_names}
    y_vals = {name: -raw[name][1] for name in block_names}  # negate Y for row

    col_slots = _cluster_axis(x_vals)  # name → slot_index (0, 1, …)
    row_slots = _cluster_axis(y_vals)  # name → slot_index (0, 1, …)

    # Step 3: for each slot, compute cumulative grid offset
    # Columns: accumulate block widths + gap per slot
    col_slot_offset = _cumulative_slot_offsets(
        block_names, col_slots, dims, axis="col", gap=gap,
    )
    row_slot_offset = _cumulative_slot_offsets(
        block_names, row_slots, dims, axis="row", gap=gap,
    )

    # Step 4: build regions
    regions: Dict[str, Tuple[int, int, int, int]] = {}
    for name in block_names:
        n_rows, n_cols = dims[name]
        r0 = row_slot_offset[row_slots[name]]
        c0 = col_slot_offset[col_slots[name]]
        regions[name] = (r0, c0, r0 + n_rows, c0 + n_cols)

    return regions


def _cluster_axis(
    values: Dict[str, float],
    tolerance: float = 0.5,
) -> Dict[str, int]:
    """Assign block names to discrete slots along one axis.

    Blocks whose axis values differ by less than *tolerance* are
    assigned the same slot.  Slots are numbered 0, 1, … in increasing
    order of axis value.

    Returns
    -------
    Dict[str, int]
        ``{block_name: slot_index}``.
    """
    sorted_names = sorted(values, key=lambda n: values[n])
    slots: Dict[str, int] = {}
    current_slot = 0
    prev_val: Optional[float] = None
    for name in sorted_names:
        v = values[name]
        if prev_val is not None and abs(v - prev_val) > tolerance:
            current_slot += 1
        slots[name] = current_slot
        prev_val = v
    return slots


def _cumulative_slot_offsets(
    block_names: List[str],
    slots: Dict[str, int],
    dims: Dict[str, Tuple[int, int]],
    axis: str,
    gap: int,
) -> Dict[int, int]:
    """Compute the starting grid coordinate for each slot.

    For each slot, the offset is the accumulated maximum dimension of
    all preceding slots (plus gap between slots).

    Returns
    -------
    Dict[int, int]
        ``{slot_index: grid_offset}``.
    """
    # Find the max dimension per slot
    n_slots = max(slots.values()) + 1 if slots else 0
    max_dim_per_slot: Dict[int, int] = {s: 0 for s in range(n_slots)}
    for name in block_names:
        s = slots[name]
        d = dims[name][0] if axis == "row" else dims[name][1]  # n_rows or n_cols
        max_dim_per_slot[s] = max(max_dim_per_slot[s], d)

    # Accumulate
    offsets: Dict[int, int] = {}
    cursor = 0
    for s in range(n_slots):
        offsets[s] = cursor
        cursor += max_dim_per_slot[s] + (gap if s < n_slots - 1 else 0)
    return offsets


def assert_disjoint_blocks(sub_grids: Dict[str, BlockSubGrid]) -> None:
    """Assert no two blocks share any trap position.

    Raises ``AssertionError`` if any ``(row, col)`` falls within the
    grid regions of two or more blocks.
    """
    all_positions: Set[Tuple[int, int]] = set()
    for name, sg in sub_grids.items():
        r0, c0, r1, c1 = sg.grid_region
        block_positions = {
            (r, c) for r in range(r0, r1) for c in range(c0, c1)
        }
        overlap = all_positions & block_positions
        assert not overlap, (
            f"Block '{name}' overlaps with previously allocated blocks "
            f"at positions {overlap}"
        )
        all_positions |= block_positions


def _extract_block_offsets(
    qubit_allocation: "QubitAllocation",
    block_names: List[str],
) -> Optional[Dict[str, Tuple[float, ...]]]:
    """Read block offsets from ``QubitAllocation.blocks``.

    Returns ``None`` when offsets are unavailable or if all blocks
    share the same offset (so the legacy side-by-side layout is used).
    """
    if not hasattr(qubit_allocation, "blocks") or not qubit_allocation.blocks:
        return None
    offsets: Dict[str, Tuple[float, ...]] = {}
    for name in block_names:
        ba = qubit_allocation.blocks.get(name)
        if ba is None or not hasattr(ba, "offset"):
            return None  # can't determine layout — fallback
        offsets[name] = tuple(float(v) for v in ba.offset)
    # If all offsets are equal the spatial layout is degenerate — fallback
    unique = set(offsets.values())
    if len(unique) <= 1:
        return None
    return offsets


def partition_grid_for_blocks(
    qec_meta: "QECMetadata",
    qubit_allocation: "QubitAllocation",
    k: int,
    gap_cols: int = 0,
) -> Dict[str, BlockSubGrid]:
    """Create per-block ``BlockSubGrid`` objects from metadata.

    Uses ``allocate_block_regions()`` to compute disjoint regions, then
    builds ``BlockSubGrid`` objects with qubit-to-ion index mappings
    (identity mappings — the real ion assignment is deferred to
    ``map_qubits_per_block``).

    Parameters
    ----------
    qec_meta : QECMetadata
        Metadata with block allocations.
    qubit_allocation : QubitAllocation
        Unified qubit allocation from the gadget layout.
    k : int
        Trap capacity.
    gap_cols : int
        Gap columns between blocks.

    Returns
    -------
    Dict[str, BlockSubGrid]
        One ``BlockSubGrid`` per block.
    """
    block_names: List[str] = [ba.block_name for ba in qec_meta.block_allocations]
    qubits_per_block: Dict[str, int] = {}

    for ba in qec_meta.block_allocations:
        n_q = (
            len(ba.data_qubits)
            + len(ba.x_ancilla_qubits)
            + len(ba.z_ancilla_qubits)
        )
        qubits_per_block[ba.block_name] = n_q

    # Include bridge ancillas in the qubit count.  Each bridge is
    # assigned to the first of its connected blocks that exists in
    # the block list.
    #
    # Fix 5: prefer explicit ``bridge_connected_blocks`` from the
    # gadget layout (populated by GadgetLayout.add_bridge_ancilla).
    # Fall back to purpose-string heuristic only when the explicit
    # mapping is absent.
    bridge_block_map: Dict[int, str] = {}  # bridge_qubit → block_name
    _explicit_bridge_map: Dict[int, List[str]] = getattr(
        qubit_allocation, 'bridge_connected_blocks', {}
    ) or {}
    if hasattr(qubit_allocation, "bridge_ancillas"):
        for gi, _coord, purpose in qubit_allocation.bridge_ancillas:
            assigned_block: Optional[str] = None

            # --- Fix 5: data-driven lookup first ---
            if gi in _explicit_bridge_map and _explicit_bridge_map[gi]:
                for _cb in _explicit_bridge_map[gi]:
                    if _cb in block_names:
                        assigned_block = _cb
                        break

            # --- Fallback: purpose-string heuristic ---
            if assigned_block is None:
                if "zz_merge" in purpose:
                    for bn in block_names:
                        if bn in ("block_0", "block_1") or bn in ("data_block", "bell_a"):
                            assigned_block = bn
                            break
                elif "xx_merge" in purpose:
                    for bn in block_names:
                        if bn in ("block_1", "block_2") or bn in ("bell_a", "bell_b"):
                            assigned_block = bn
                            break

            if assigned_block is None and block_names:
                assigned_block = block_names[0]
            if assigned_block is not None:
                bridge_block_map[gi] = assigned_block
                qubits_per_block[assigned_block] = (
                    qubits_per_block.get(assigned_block, 0) + 1
                )

    regions = allocate_block_regions(
        block_names=block_names,
        qubits_per_block=qubits_per_block,
        k=k,
        gap_cols=gap_cols,
        block_offsets=_extract_block_offsets(qubit_allocation, block_names),
    )

    sub_grids: Dict[str, BlockSubGrid] = {}
    for ba in qec_meta.block_allocations:
        r0, c0, r1, c1 = regions[ba.block_name]
        all_qubits = (
            list(ba.data_qubits)
            + list(ba.x_ancilla_qubits)
            + list(ba.z_ancilla_qubits)
        )
        # Add bridge ancillas assigned to this block
        for gi, assigned_bn in bridge_block_map.items():
            if assigned_bn == ba.block_name:
                all_qubits.append(gi)

        # 1-based qubit→ion mapping (ion 0 is the empty sentinel in layout
        # arrays).  Matches the global qubit_to_ion = {q: q+1} built in
        # route_full_experiment and used by _create_ions_from_allocation.
        q2i: Dict[int, int] = {q: q + 1 for q in all_qubits}

        sub_grids[ba.block_name] = BlockSubGrid(
            block_name=ba.block_name,
            grid_region=(r0, c0, r1, c1),
            n_rows=r1 - r0,
            n_cols=c1 - c0,
            ion_indices=all_qubits,  # will be overwritten by map_qubits_per_block
            qubit_to_ion=q2i,
        )

    # Verify disjointness
    assert_disjoint_blocks(sub_grids)
    return sub_grids


def compute_gadget_grid_size(
    qec_metadata: "QECMetadata",
    qubit_allocation: "QubitAllocation",
    k: int,
    gap_cols: int = 0,
) -> Tuple[int, int]:
    """Compute WISE grid dimensions for a multi-block gadget experiment.

    Uses :func:`partition_grid_for_blocks` to allocate disjoint
    per-block sub-grid regions and returns the grid size that
    encloses them all.

    Parameters
    ----------
    qec_metadata : QECMetadata
        Metadata with block allocations (from
        ``FaultTolerantGadgetExperiment.qec_metadata``).
    qubit_allocation : QubitAllocation
        Unified qubit allocation (from
        ``FaultTolerantGadgetExperiment._unified_allocation``).
    k : int
        Trap capacity (ions per trap).
    gap_cols : int
        Gap columns between blocks.

    Returns
    -------
    m_traps : int
        Number of trap *columns* (``m`` parameter for
        :class:`QCCDWiseArch`).
    n_traps : int
        Number of trap *rows* (``n`` parameter for
        :class:`QCCDWiseArch`).
    """
    sub_grids = partition_grid_for_blocks(
        qec_metadata, qubit_allocation, k, gap_cols=gap_cols,
    )
    max_r = max(sg.grid_region[2] for sg in sub_grids.values())
    max_c = max(sg.grid_region[3] for sg in sub_grids.values())
    return max_c, max_r  # (m_traps, n_traps)


def map_qubits_per_block(
    sub_grids: Dict[str, BlockSubGrid],
    measurement_ions_per_block: Dict[str, List[Any]],
    data_ions_per_block: Dict[str, List[Any]],
    k: int,
) -> Dict[str, np.ndarray]:
    """Run ``regularPartition`` + ``hillClimbOnArrangeClusters`` per block.

    Each block is mapped to its own allocated sub-grid region. Different
    blocks NEVER share traps.  This is the key function for guaranteeing
    disjoint block layouts (§3.1.1 of the spec).

    Parameters
    ----------
    sub_grids : Dict[str, BlockSubGrid]
        Per-block sub-grid allocations (from ``partition_grid_for_blocks``).
    measurement_ions_per_block : Dict[str, List[Ion]]
        Measurement (ancilla) ions per block.
    data_ions_per_block : Dict[str, List[Ion]]
        Data ions per block.
    k : int
        Trap capacity.

    Returns
    -------
    Dict[str, np.ndarray]
        Per-block initial layout arrays (``n_rows × (n_cols * k)``
        ion-index arrangement, with 0 for empty).  Also updates
        ``sub_grids[block_name].initial_layout`` and
        ``sub_grids[block_name].ion_indices`` in place.
    """
    from ..compiler.qccd_qubits_to_ions import (
        regularPartition,
        hillClimbOnArrangeClusters,
    )

    layouts: Dict[str, np.ndarray] = {}

    for block_name, sg in sub_grids.items():
        r0, c0, r1, c1 = sg.grid_region
        block_rows = r1 - r0
        block_cols = c1 - c0

        m_ions = measurement_ions_per_block.get(block_name, [])
        d_ions = data_ions_per_block.get(block_name, [])

        if not m_ions and not d_ions:
            # Empty block — no ions to place
            layout = np.zeros((block_rows, block_cols * k), dtype=int)
            layouts[block_name] = layout
            sg.initial_layout = layout
            sg.ion_indices = []
            continue

        # Partition THIS block's ions into clusters
        max_clusters = block_rows * block_cols
        clusters = regularPartition(
            m_ions,
            d_ions,
            k,
            isWISEArch=True,
            maxClusters=max_clusters,
        )

        # Map clusters to positions within THIS block's sub-grid only
        # Note: gridPositions uses (col, row) format matching WISE convention
        block_grid_pos = [
            (c, r) for r in range(block_rows) for c in range(block_cols)
        ]
        grid_positions = hillClimbOnArrangeClusters(
            clusters, allGridPos=block_grid_pos
        )

        # ------------------------------------------------------------------
        # Fix 6: Swap bridge-containing clusters toward their preferred
        # grid edge.  Bridge ancillas tagged with ``_preferred_edge`` in
        # ``_create_ions_from_allocation`` should sit on the block boundary
        # facing the adjacent block so merge MS gates need minimal transport.
        # ------------------------------------------------------------------
        grid_positions = list(grid_positions)  # ensure mutable
        _bridge_prefs: Dict[int, str] = {}
        for _ci, (_cions, _) in enumerate(clusters):
            for _ion in _cions:
                _pref = getattr(_ion, '_preferred_edge', None)
                if _pref is not None:
                    _bridge_prefs[_ci] = _pref
                    break

        if _bridge_prefs:
            for _bc_idx, _pref_edge in _bridge_prefs.items():
                _bc_col, _bc_row = grid_positions[_bc_idx]
                _tgt_row, _tgt_col = _bc_row, _bc_col

                if _pref_edge == "bottom":
                    _tgt_row = block_rows - 1
                elif _pref_edge == "top":
                    _tgt_row = 0
                elif _pref_edge == "right":
                    _tgt_col = block_cols - 1
                elif _pref_edge == "left":
                    _tgt_col = 0

                if (_bc_col, _bc_row) == (_tgt_col, _tgt_row):
                    continue  # already on preferred edge

                # Find a non-bridge cluster on the target edge to swap with.
                # Prefer the one closest to the bridge cluster's position.
                _best_swap = None
                _best_dist = float('inf')
                for _oi, (_oc, _or) in enumerate(grid_positions):
                    if _oi in _bridge_prefs:
                        continue  # don't displace another bridge
                    _on_edge = False
                    if _pref_edge in ("bottom", "top") and _or == _tgt_row:
                        _on_edge = True
                    elif _pref_edge in ("left", "right") and _oc == _tgt_col:
                        _on_edge = True
                    if _on_edge:
                        _d = abs(_oc - _bc_col) + abs(_or - _bc_row)
                        if _d < _best_dist:
                            _best_dist = _d
                            _best_swap = _oi

                if _best_swap is not None:
                    grid_positions[_bc_idx], grid_positions[_best_swap] = (
                        grid_positions[_best_swap], grid_positions[_bc_idx]
                    )

        # Build block-local arrangement array
        layout = np.zeros((block_rows, block_cols * k), dtype=int)
        ion_indices: List[int] = []

        for cluster_idx, (col, row) in enumerate(grid_positions):
            cluster_ions = clusters[cluster_idx][0]
            for ion_pos, ion in enumerate(cluster_ions):
                grid_col = col * k + ion_pos
                if row < block_rows and grid_col < block_cols * k:
                    layout[row, grid_col] = ion.idx
                    ion_indices.append(ion.idx)

        layouts[block_name] = layout
        sg.initial_layout = layout
        sg.ion_indices = ion_indices

    return layouts


# =====================================================================
# Step 4: Ion-Return BT Constraints
# =====================================================================

def build_ion_return_bt_for_patch_and_route(
    initial_layout: np.ndarray,
    num_rounds: int = 1,
    active_ions: Optional[Collection[int]] = None,
) -> List[Dict[Tuple[int, int], Dict[int, Tuple[int, int]]]]:
    """Build BT structure for ``_patch_and_route`` with ion-return pins.

    ``_patch_and_route`` expects ``BTs`` as a list indexed by round,
    where each entry is a dict keyed by ``(cycle_idx, tiling_idx)``
    mapping to ``(bt_map, solved_pairs)`` or just ``bt_map``.

    For ion-return, the pin map is placed **only on the final round**
    — an extra trailing empty round appended by the caller (no MS
    pairs).  Earlier rounds get empty dicts so the SAT solver can
    freely reconfigure ions for MS gates.

    Parameters
    ----------
    initial_layout : np.ndarray
        ``n_rows × n_cols`` arrangement array.
    num_rounds : int
        Number of MS-carrying rounds.  The returned list has length
        ``num_rounds + 1``.
    active_ions : Collection[int] or None
        If provided, only pin these ions in the return round.
        Spectator ions (those not in *active_ions*) are left free,
        giving the SAT solver additional degrees of freedom.  When
        ``None``, all non-zero ions in the layout are pinned (legacy
        behaviour).

    Returns
    -------
    List[Dict[Tuple[int, int], Dict[int, Tuple[int, int]]]]
        Length ``num_rounds + 1``.  Empty dicts for MS-carrying
        rounds; ``{(0, 0): ion_pin_map}`` for the trailing round.

    Notes
    -----
    The caller **must** also append an empty ``[]`` to ``P_arr``
    so that the trailing round has no MS pairs.
    """
    active_set = set(active_ions) if active_ions is not None else None
    bt_map: Dict[int, Tuple[int, int]] = {}
    n_rows, n_cols = initial_layout.shape
    for r in range(n_rows):
        for c in range(n_cols):
            ion_idx = int(initial_layout[r, c])
            if ion_idx != 0:
                if active_set is None or ion_idx in active_set:
                    bt_map[ion_idx] = (r, c)

    # Empty BT for MS-carrying rounds, pin map on trailing return round
    return [dict() for _ in range(num_rounds)] + [{(0, 0): dict(bt_map)}]


def _compute_return_reconfig(
    final_layout: np.ndarray,
    target_layout: np.ndarray,
    arch,
    subgridsize,
    base_pmax_in: int = 1,
    stop_event=None,
    max_inner_workers=None,
    progress_callback=None,
    allow_heuristic_fallback: bool = False,
    use_heuristic_directly: bool = False,
    max_sat_time: Optional[float] = None,
    max_rc2_time: Optional[float] = None,
    solver_params: Optional[Any] = None,
) -> list:
    """Compute return-round reconfiguration.

    When ``use_heuristic_directly=True``, skips SAT entirely and produces
    a single RoutingStep with ``schedule=None``.  This causes
    ``physicalOperation`` to use the deterministic Phase B/C/D heuristic
    (odd-even transposition sort) which always succeeds.  This is the
    correct behaviour when ``heuristic_route_back=True`` — the plan
    phase (MS routing) is complete and all that remains is returning
    ions to start positions, which does not require SAT.

    When ``use_heuristic_directly=False``, uses
    ``_rebuild_schedule_for_layout`` for SAT-based convergence (as before).

    Parameters
    ----------
    final_layout : np.ndarray
        Layout after all MS rounds have been routed.
    target_layout : np.ndarray
        Desired ion positions (typically the initial EC layout).
    arch : QCCDWiseArch
        Grid geometry.
    subgridsize : tuple
        Patch dimensions for SAT tiling.
    base_pmax_in : int
        Base P_max for SAT solver.
    use_heuristic_directly : bool
        If True, skip SAT and use heuristic (schedule=None) immediately.
    stop_event, max_inner_workers : optional
        Threading controls.

    Returns
    -------
    list
        List of ``RoutingStep`` objects for the return reconfiguration.
        Empty list if layouts already match.

    """
    from ..compiler.qccd_WISE_ion_route import (
        _rebuild_schedule_for_layout,
        RoutingStep,
    )

    if np.array_equal(final_layout, target_layout):
        return []

    _logger = logging.getLogger("wise.qccd.gadget_routing")
    _mismatch = int(np.sum(final_layout != target_layout))
    _logger.debug(
        "[ReturnRound] computing separate return reconfig "
        "(%d non-matching cells, use_heuristic_directly=%s)",
        _mismatch, use_heuristic_directly,
    )

    # ── Fast path: skip SAT entirely when heuristic is requested ──
    # The plan phase (MS routing) is already complete; all that remains
    # is moving ions back to their start positions.  The heuristic
    # (odd-even transposition sort) is O(m+n) passes and always
    # succeeds for any permutation — no need to invoke SAT.
    if use_heuristic_directly:
        _logger.info(
            "[ReturnRound] skipping SAT — using heuristic directly "
            "(%d non-matching cells)",
            _mismatch,
        )
        return [RoutingStep(
            layout_after=np.array(target_layout, copy=True),
            schedule=None,
            solved_pairs=[],
            ms_round_index=-1,
            from_cache=False,
            tiling_meta=(0, 0),
            can_merge_with_next=False,
            is_initial_placement=False,
            is_layout_transition=True,
            reconfig_context="return_round",
            layout_before=np.array(final_layout, copy=True),
        )]

    # Route-back / cache-replay context: heuristic fallback is acceptable.
    # If SAT cannot fully converge, _rebuild_schedule_for_layout
    # appends a final snapshot with schedule=None, which causes
    # physicalOperation to use the heuristic odd-even sort instead.
    snaps = _rebuild_schedule_for_layout(
        final_layout.copy(),
        arch,
        target_layout,
        subgridsize=subgridsize,
        base_pmax_in=base_pmax_in,
        stop_event=stop_event,
        max_inner_workers=max_inner_workers or 1,
        progress_callback=progress_callback,
        allow_heuristic_fallback=allow_heuristic_fallback,
        max_sat_time=max_sat_time,
        max_rc2_time=max_rc2_time,
        solver_params=solver_params,
    )

    steps = []
    _prev_layout = np.array(final_layout, copy=True)
    for i, (layout, schedule, _pairs) in enumerate(snaps):
        steps.append(RoutingStep(
            layout_after=np.array(layout, copy=True),
            schedule=schedule,
            solved_pairs=[],
            ms_round_index=-1,
            from_cache=False,
            tiling_meta=(i, 0),
            can_merge_with_next=False,
            is_initial_placement=False,
            is_layout_transition=True,
            reconfig_context="return_round",
            layout_before=_prev_layout,
        ))
        _prev_layout = np.array(layout, copy=True)
    return steps


# =====================================================================
# Integration Layer: Full Experiment Result
# =====================================================================

@dataclass
class PhaseResult:
    """Routing result for one temporal phase.

    Uniform interface for the full-experiment orchestrator.

    Attributes
    ----------
    phase_index : int
        Index into ``QECMetadata.phases``.
    phase_type : str
        ``"ec"``, ``"gadget"``, or ``"transition"``.
    layouts : List[np.ndarray]
        Sequence of layout arrays produced during this phase.
    schedule : List[List[Dict[str, Any]]]
        Per-step reconfiguration schedule.
    exec_time : float
        Estimated execution time (μs) for this phase.
    reconfig_time : float
        Subset of exec_time spent on reconfigurations.
    from_cache : bool
        True if this phase was served from the round cache.
    """
    phase_index: int = 0
    phase_type: str = ""
    layouts: List[np.ndarray] = field(default_factory=list)
    schedule: List[List[Dict[str, Any]]] = field(default_factory=list)
    exec_time: float = 0.0
    reconfig_time: float = 0.0
    from_cache: bool = False


@dataclass
class FullExperimentResult:
    """Complete routing result for a fault-tolerant gadget experiment.

    Produced by ``route_full_experiment()`` — the main integration point
    that replaces the old monolithic ``_route_and_simulate()`` pipeline.

    Attributes
    ----------
    phase_results : List[PhaseResult]
        One result per routed phase.
    total_schedule : List[List[Dict[str, Any]]]
        Concatenated schedules from all phases (for timing computation).
    total_exec_time : float
        Total estimated execution time (μs).
    total_reconfig_time : float
        Total reconfiguration time across all phases (μs).
    sub_grids : Dict[str, BlockSubGrid]
        Per-block sub-grid allocations.
    per_ion_heating : Dict[int, float]
        Per-ion accumulated motional heating (quanta).
    cached_phases : int
        Number of phases served from cache.
    total_phases : int
        Total number of routed phases.
    ms_rounds_routed : int
        Total MS rounds actually routed by SAT solver.
    ms_rounds_replayed : int
        Total MS rounds served via cache replay.
    """
    phase_results: List[PhaseResult] = field(default_factory=list)
    total_schedule: List[List[Dict[str, Any]]] = field(default_factory=list)
    total_exec_time: float = 0.0
    total_reconfig_time: float = 0.0
    sub_grids: Dict[str, BlockSubGrid] = field(default_factory=dict)
    per_ion_heating: Dict[int, float] = field(default_factory=dict)
    cached_phases: int = 0
    total_phases: int = 0
    ms_rounds_routed: int = 0
    ms_rounds_replayed: int = 0


# =====================================================================
# Integration Layer: Schedule Timing Computation
# =====================================================================

def compute_schedule_timing(
    schedules: List[List[Dict[str, Any]]],
    n_rows: int = 0,
    n_cols: int = 0,
    k: int = 2,
    phase_type: str = "",
    num_stabilizer_rounds: int = 0,
) -> Tuple[float, float, Dict[int, float]]:
    """Compute total execution time and per-ion heating from schedules.

    Mirrors the timing logic in ``GlobalReconfigurations._runOddEvenReconfig``
    and ``qccd_SAT_WISE_odd_even_sorter._estimate_reconfig_time`` for the
    reconfiguration component, then adds gate-level overhead that the old
    pipeline computes via ``paralleliseOperationsWithBarriers``.

    The old pipeline's ``exec_time`` (``scheduled.total_duration``) includes
    **all** physical operations: transport (reconfigurations), MS gates,
    single-qubit gates, measurements, resets, and recooling.  The routing
    schedule only captures the transport component, so we analytically
    account for the remaining operations per phase:

    * **EC phases** (stabilizer rounds): per stabilizer round adds
      measurement + reset + recooling + single-qubit basis-change gates.
    * **Gadget phases** (transversal CNOT): per gadget step adds minimal
      single-qubit overhead for basis changes.
    * All phases: one MS gate execution per reconfiguration round.

    Parameters
    ----------
    schedules : List[List[Dict[str, Any]]]
        Per-round schedule passes.  Each round is a list of pass dicts
        with keys ``"phase"`` (``"H"`` or ``"V"``), ``"h_swaps"``,
        ``"v_swaps"``.
    n_rows : int
        Grid row count (for heating calculation).
    n_cols : int
        Full grid column count including trap capacity factor.
    k : int
        Trap capacity.
    phase_type : str
        ``"ec"`` or ``"gadget"`` — determines which overhead terms apply.
    num_stabilizer_rounds : int
        Number of logical stabilizer rounds in this phase (EC phases
        only).  Each round contributes measurement + reset + recooling
        overhead.

    Returns
    -------
    Tuple[float, float, Dict[int, float]]
        ``(total_exec_time, total_reconfig_time, per_ion_heating)``
        Times are in seconds.  Heating is in quanta.
    """
    from .qccd_operations import (
        Move, Merge, CrystalRotation, Split, JunctionCrossing,
    )

    row_swap_time = (
        Move.MOVING_TIME
        + Merge.MERGING_TIME
        + CrystalRotation.ROTATION_TIME
        + Split.SPLITTING_TIME
        + Move.MOVING_TIME
    )
    row_swap_heating = (
        Move.MOVING_TIME * Move.HEATING_RATE
        + Merge.MERGING_TIME * Merge.HEATING_RATE
        + CrystalRotation.ROTATION_TIME * CrystalRotation.HEATING_RATE
        + Split.SPLITTING_TIME * Split.HEATING_RATE
        + Move.MOVING_TIME * Move.HEATING_RATE
    )
    col_swap_time = (2 * JunctionCrossing.CROSSING_TIME) + (
        4 * JunctionCrossing.CROSSING_TIME + Move.MOVING_TIME
    ) * 2
    col_swap_heating = (
        6 * JunctionCrossing.CROSSING_TIME * JunctionCrossing.HEATING_RATE
        + Move.MOVING_TIME * Move.HEATING_RATE
    )

    total_reconfig_time = 0.0
    per_ion_heating: Dict[int, float] = {}

    for round_passes in schedules:
        if not round_passes:
            continue
        # Fixed overhead per reconfig: initial split
        round_time = Split.SPLITTING_TIME
        for pass_info in round_passes:
            phase = pass_info.get("phase", "")
            if phase == "H":
                h_swaps = pass_info.get("h_swaps", [])
                if h_swaps:
                    round_time += row_swap_time
                    # Accumulate per-ion heating for swapped ions
                    for swap in h_swaps:
                        if isinstance(swap, (list, tuple)) and len(swap) >= 2:
                            # swap is (row, col) — we'd need ion IDs for per-ion tracking
                            # For now, aggregate total only
                            pass
            elif phase == "V":
                v_swaps = pass_info.get("v_swaps", [])
                if v_swaps:
                    round_time += col_swap_time

        total_reconfig_time += round_time

    # ----- Gate-level overhead (beyond reconfiguration) -----
    from .physics import DEFAULT_CALIBRATION
    cal = DEFAULT_CALIBRATION

    # MS gate execution time: one parallel MS batch per reconfiguration round
    total_ms_time = cal.ms_gate_time * len(schedules)

    # Per-stabilizer-round overhead for EC phases:
    #   - measurement of all ancilla qubits   (parallel batch)
    #   - reset of measured ancillas           (parallel batch)
    #   - sympathetic recooling                (parallel batch)
    #   - 2 layers of single-qubit gates       (H basis-change pre + post)
    # These mirror what paralleliseOperationsWithBarriers produces in
    # the old pipeline — each is one WISE batch (global barrier).
    if phase_type == "ec" and num_stabilizer_rounds > 0:
        per_stab_round_overhead = (
            cal.measurement_time          # ancilla measurement   (400 μs)
            + cal.reset_time              # ancilla reset         ( 50 μs)
            + cal.recool_time             # sympathetic recooling (400 μs)
            + 2 * cal.single_qubit_gate_time  # H pre + H post   (  10 μs)
        )
        gate_overhead = num_stabilizer_rounds * per_stab_round_overhead
    elif phase_type == "gadget":
        # Gadget phases (transversal CNOT): only basis-change single-qubit
        # gates before/after the MS layers.  No measurement/reset.
        gate_overhead = 2 * cal.single_qubit_gate_time
    else:
        gate_overhead = 0.0

    total_exec_time = total_reconfig_time + total_ms_time + gate_overhead

    return total_exec_time, total_reconfig_time, per_ion_heating


# =====================================================================
# Integration Layer: Phase Decomposition
# =====================================================================

def decompose_into_phases(
    qec_meta: "QECMetadata",
    gadget: "Gadget",
    qubit_allocation: "QubitAllocation",
    sub_grids: Dict[str, BlockSubGrid],
    qubit_to_ion: Dict[int, int],
    k: int,
) -> List[PhaseRoutingPlan]:
    """Convert ``QECMetadata.phases`` into a sequence of routing plans.

    For each phase in ``qec_meta.phases``:
      - Determine ``phase_type``: ``"ec"``, ``"gadget"``, or ``"transition"``
      - Identify interacting and idle blocks
      - Extract MS pairs (EC → per-block stabilizer schedule; gadget → transversal)
      - Set caching / dedup flags

    Parameters
    ----------
    qec_meta : QECMetadata
        Metadata from ``FaultTolerantGadgetExperiment.qec_metadata``.
    gadget : Gadget
        The gadget object (for ``derive_gadget_ms_pairs``).
    qubit_allocation : QubitAllocation
        Full qubit allocation from the experiment layout.
    sub_grids : Dict[str, BlockSubGrid]
        Per-block sub-grid allocations.
    qubit_to_ion : Dict[int, int]
        Global qubit → ion mapping.
    k : int
        Trap capacity.

    Returns
    -------
    List[PhaseRoutingPlan]
        One plan per phase, in temporal order.
    """
    all_block_names = [ba.block_name for ba in qec_meta.block_allocations]
    plans: List[PhaseRoutingPlan] = []

    # Track round signatures for cache dedup across phases
    seen_signatures: Dict[Tuple, int] = {}

    # Gadget-phase-local counter: derive_gadget_ms_pairs expects a
    # counter that starts at 0 for the first gadget phase and increments
    # by 1 for each subsequent gadget phase (NOT the global phase index).
    _gadget_phase_counter = 0

    for i, phase in enumerate(qec_meta.phases):
        phase_type = phase.phase_type or ""
        # Distinguish None (use all blocks) from [] (no active blocks).
        # The old ``or`` treated [] as falsy and fell back to all blocks,
        # which generated ghost EC routing rounds for phases like
        # stabilizer_round_pre with active_blocks=[].
        active_blocks = (
            phase.active_blocks
            if phase.active_blocks is not None
            else list(all_block_names)
        )
        idle_blocks = [b for b in all_block_names if b not in active_blocks]

        # Skip non-routing phases (init, measure)
        if phase_type in ("init", "measure", ""):
            continue

        # Match EC-like phase types (stabilizer_round, stabilizer_round_pre,
        # stabilizer_round_post, final_round, ec, etc.)
        _is_ec = _is_ec_phase_type(phase_type)

        if _is_ec:
            # EC phase: derive MS pairs per active block from metadata
            ms_pairs_one_round: List[List[Tuple[int, int]]] = []
            for block_name in active_blocks:
                if block_name in sub_grids:
                    block_q2i = sub_grids[block_name].qubit_to_ion
                    block_ms = derive_ms_pairs_from_metadata(
                        qec_meta, block_q2i, block_name
                    )
                    # Merge all blocks' pairs per round (they don't overlap)
                    for r_idx, round_pairs in enumerate(block_ms):
                        while len(ms_pairs_one_round) <= r_idx:
                            ms_pairs_one_round.append([])
                        ms_pairs_one_round[r_idx].extend(round_pairs)

            # ── Fix 13: Repeat EC pairs for num_rounds > 1 ──
            # derive_ms_pairs_from_metadata returns ONE stabilizer round's
            # worth of pairs.  If phase.num_rounds > 1, we need to repeat
            # the pattern num_rounds times.
            ec_num_rounds = phase.num_rounds if phase.num_rounds > 0 else 1
            ms_pairs_all_rounds: List[List[Tuple[int, int]]] = []
            for _ in range(ec_num_rounds):
                for round_pairs in ms_pairs_one_round:
                    ms_pairs_all_rounds.append(list(round_pairs))

            # Build round signature for cache dedup
            sig = None
            if ms_pairs_all_rounds:
                sig = tuple(
                    tuple(sorted(rp)) for rp in ms_pairs_all_rounds
                )

            identical_to = None
            if sig is not None and sig in seen_signatures:
                identical_to = seen_signatures[sig]
            elif sig is not None:
                seen_signatures[sig] = i

            plan = PhaseRoutingPlan(
                phase_type="ec",
                phase_index=i,
                interacting_blocks=list(active_blocks),
                idle_blocks=idle_blocks,
                all_blocks=list(all_block_names),
                ms_pairs_per_round=ms_pairs_all_rounds,
                num_rounds=ec_num_rounds,
                is_cached=(identical_to is not None),
                grid_region=(0, 0, 0, 0),  # filled by caller if needed
                round_signature=sig,
                identical_to_phase=identical_to,
            )
            plans.append(plan)

        elif phase_type == "gadget":
            # Gadget phase: derive MS pairs.
            # Prefer gadget.get_phase_pairs() (N-block safe, handles bridge
            # qubits, multi-round merge phases, and no-MS phases).
            # Fall back to derive_gadget_ms_pairs() transversal heuristic.
            ms_pairs: List[List[Tuple[int, int]]] = []
            _has_phase_pairs = (
                hasattr(gadget, "get_phase_pairs")
                and callable(getattr(gadget, "get_phase_pairs", None))
            )
            if _has_phase_pairs:
                try:
                    raw_pairs = gadget.get_phase_pairs(
                        _gadget_phase_counter, qubit_allocation,
                    )
                    # Convert qubit indices → ion indices
                    _dropped = 0
                    for round_pairs in raw_pairs:
                        ion_round = []
                        for ctrl_q, tgt_q in round_pairs:
                            c_ion = qubit_to_ion.get(ctrl_q)
                            t_ion = qubit_to_ion.get(tgt_q)
                            if c_ion is not None and t_ion is not None:
                                ion_round.append((c_ion, t_ion))
                            else:
                                _dropped += 1
                        if ion_round:
                            ms_pairs.append(ion_round)
                    if _dropped > 0:
                        logger.warning(
                            "decompose_into_phases: gadget phase %d dropped "
                            "%d MS pair(s) – qubit(s) missing from "
                            "qubit_to_ion (bridge ancillas included?)",
                            _gadget_phase_counter, _dropped,
                        )
                except NotImplementedError:
                    _has_phase_pairs = False

            if not _has_phase_pairs:
                # Fallback: old transversal heuristic (2-block only)
                if len(active_blocks) >= 2:
                    ms_pairs = derive_gadget_ms_pairs(
                        gadget, _gadget_phase_counter, qubit_allocation,
                        active_blocks, qubit_to_ion,
                    )
                # else: single-block phase with no get_phase_pairs → empty

            _gadget_phase_counter += 1

            # ── Interleave EC stabiliser rounds within gadget phase ──
            # CSS-surgery stim circuits interleave bridge CX with EC
            # stabiliser CX: bridge_R1 → EC(N layers) → bridge_R2 → …
            # Without EC rounds in the routing plan the ions won't be
            # co-located for the in-block stabiliser MS gates and they
            # will never execute.
            _round_labels: List[str] = []
            if ms_pairs:
                ec_pairs_all: List[List[Tuple[int, int]]] = []
                for _bn in active_blocks:
                    if _bn in sub_grids:
                        _bq2i = sub_grids[_bn].qubit_to_ion
                        _bms = derive_ms_pairs_from_metadata(
                            qec_meta, _bq2i, _bn
                        )
                        for _ri, _rp in enumerate(_bms):
                            while len(ec_pairs_all) <= _ri:
                                ec_pairs_all.append([])
                            ec_pairs_all[_ri].extend(_rp)
                if ec_pairs_all:
                    _interleaved: List[List[Tuple[int, int]]] = []
                    for _gadget_round in ms_pairs:
                        # Fix 7: split shared-ion bridge rounds into
                        # disjoint sub-rounds before interleaving with
                        # EC rounds.  The SAT solver requires all pairs
                        # within a round to be co-located simultaneously.
                        _sub_rounds = _split_shared_ion_rounds(
                            [_gadget_round],
                        )
                        # Fix C: Merge bridge sub-rounds with EC rounds
                        # into combined routing rounds.  Bridge pairs are
                        # inter-block and EC pairs are intra-block, so
                        # they typically operate on disjoint ions and can
                        # share a single SAT solve.  Ion conflicts (if
                        # any) are handled by the SAT solver naturally.
                        # This reduces rounds per merge cycle from
                        # (n_sub + n_ec) to max(n_sub, n_ec).
                        _n_sub = len(_sub_rounds)
                        _n_ec = len(ec_pairs_all)
                        _n_combined = max(_n_sub, _n_ec)
                        for _ci in range(_n_combined):
                            _combined: List[Tuple[int, int]] = []
                            _lbl = "ec"
                            if _ci < _n_sub:
                                _combined.extend(_sub_rounds[_ci])
                                _lbl = "bridge" if _ci >= _n_ec else "combined"
                            if _ci < _n_ec:
                                _combined.extend(ec_pairs_all[_ci])
                            if _combined:
                                _interleaved.append(_combined)
                                _round_labels.append(_lbl)
                    ms_pairs = _interleaved
                else:
                    _round_labels = ["bridge"] * len(ms_pairs)
            else:
                # No MS pairs → single-qubit phase (split/measurement)
                _round_labels = []

            num_rounds = max(len(ms_pairs), 1)

            # Fix B: Compute round_signature for gadget phases so that
            # identical gadget phases (e.g. repeated merge rounds in CSS
            # Surgery) can be cached and replayed by
            # route_full_experiment_as_steps instead of freshly SAT-routed.
            _gadget_sig = None
            _gadget_identical_to = None
            if ms_pairs:
                _gadget_sig = tuple(
                    tuple(sorted(tuple(sorted(p)) for p in rnd))
                    for rnd in ms_pairs
                )
                if _gadget_sig in seen_signatures:
                    _gadget_identical_to = seen_signatures[_gadget_sig]
                else:
                    seen_signatures[_gadget_sig] = i

            plan = PhaseRoutingPlan(
                phase_type="gadget",
                phase_index=i,
                interacting_blocks=list(active_blocks),
                idle_blocks=idle_blocks,
                all_blocks=list(all_block_names),
                ms_pairs_per_round=ms_pairs,
                num_rounds=num_rounds,
                is_cached=(_gadget_identical_to is not None),
                grid_region=(0, 0, 0, 0),
                round_signature=_gadget_sig,
                identical_to_phase=_gadget_identical_to,
                round_labels=_round_labels,
            )
            plans.append(plan)

    return plans


# =====================================================================
# Integration Layer: Ion Creation from QubitAllocation
# =====================================================================

def _create_ions_from_allocation(
    qec_meta: "QECMetadata",
    qubit_allocation: "QubitAllocation",
) -> Tuple[Dict[str, List[Any]], Dict[str, List[Any]]]:
    """Create measurement and data Ion objects from block allocations.

    Uses the code's coordinate system (via ``get_code_coords``) and the
    block's spatial offset to assign positions to each ion.  This
    mirrors the QUBIT_COORDS emission logic in
    ``QubitAllocation.emit_qubit_coords`` and the ion classification
    in ``TrappedIonCompiler._decompose_and_classify``.

    Returns
    -------
    Tuple[Dict[str, List[Ion]], Dict[str, List[Ion]]]
        ``(measurement_ions_per_block, data_ions_per_block)``
    """
    from ..utils.qccd_nodes import QubitIon
    from qectostim.gadgets.coordinates import get_code_coords

    measurement_ions_per_block: Dict[str, List[Any]] = {}
    data_ions_per_block: Dict[str, List[Any]] = {}

    for ba in qec_meta.block_allocations:
        block_name = ba.block_name
        block_alloc = qubit_allocation.blocks.get(block_name)
        if block_alloc is None:
            measurement_ions_per_block[block_name] = []
            data_ions_per_block[block_name] = []
            continue

        code = block_alloc.code
        offset = block_alloc.offset

        # Get code coordinates
        data_coords, x_stab_coords, z_stab_coords = get_code_coords(code)

        m_ions: List[Any] = []
        d_ions: List[Any] = []

        # Data qubits → data ions
        # Ion indices are 1-based (global_idx + 1) because
        # layout arrays use 0 as the "empty slot" sentinel.
        # This matches the Trap constructor convention.
        for i, global_idx in enumerate(block_alloc.data_range):
            if i < len(data_coords):
                coord = data_coords[i]
                x = int(float(coord[0]) + offset[0])
                y = int(float(coord[1]) + offset[1]) if len(coord) > 1 else int(offset[1])
            else:
                x = int(float(i) + offset[0])
                y = int(offset[1])

            ion = QubitIon("#7fc7af", label="D")
            ion.set(global_idx + 1, x, y)
            d_ions.append(ion)

        # X ancilla qubits → measurement ions
        for i, global_idx in enumerate(block_alloc.x_anc_range):
            if i < len(x_stab_coords):
                coord = x_stab_coords[i]
                x = int(float(coord[0]) + offset[0])
                y = int(float(coord[1]) + offset[1]) if len(coord) > 1 else int(offset[1])
            else:
                x = int(float(i) + offset[0])
                y = int(1.0 + offset[1])

            ion = QubitIon("#e8927c", label="M")
            ion.set(global_idx + 1, x, y)
            m_ions.append(ion)

        # Z ancilla qubits → measurement ions
        for i, global_idx in enumerate(block_alloc.z_anc_range):
            if i < len(z_stab_coords):
                coord = z_stab_coords[i]
                x = int(float(coord[0]) + offset[0])
                y = int(float(coord[1]) + offset[1]) if len(coord) > 1 else int(offset[1])
            else:
                x = int(float(i) + offset[0])
                y = int(-1.0 + offset[1])

            ion = QubitIon("#e8927c", label="M")
            ion.set(global_idx + 1, x, y)
            m_ions.append(ion)

        measurement_ions_per_block[block_name] = m_ions
        data_ions_per_block[block_name] = d_ions

    # ---- Bridge ancilla ions ----
    # Bridge ancillas connect blocks during merge phases.  Assign each
    # to the first of its connected blocks that exists.  Position them
    # at the boundary of that block *toward* the adjacent block so the
    # first MS round doesn't require long-range transport.
    if hasattr(qubit_allocation, "bridge_ancillas") and qubit_allocation.bridge_ancillas:
        _bridge_connected = getattr(qubit_allocation, "bridge_connected_blocks", {}) or {}
        block_names_set = set(measurement_ions_per_block.keys())

        for gi, coord, purpose in qubit_allocation.bridge_ancillas:
            # Determine which block this bridge belongs to
            assigned_block: Optional[str] = None
            if gi in _bridge_connected:
                for cb in _bridge_connected[gi]:
                    if cb in block_names_set:
                        assigned_block = cb
                        break
            if assigned_block is None:
                # Fallback: purpose-string heuristic
                if "zz_merge" in purpose:
                    for bn in ("block_0", "block_1"):
                        if bn in block_names_set:
                            assigned_block = bn
                            break
                elif "xx_merge" in purpose:
                    for bn in ("block_1", "block_2"):
                        if bn in block_names_set:
                            assigned_block = bn
                            break
            if assigned_block is None and block_names_set:
                assigned_block = sorted(block_names_set)[0]
            if assigned_block is None:
                continue

            # Position: use bridge coord directly (already at boundary)
            x = int(float(coord[0])) if len(coord) > 0 else 0
            y = int(float(coord[1])) if len(coord) > 1 else 0

            # Bias toward the block boundary facing the adjacent block.
            # Find the offset of the assigned block and its neighbour.
            block_alloc = qubit_allocation.blocks.get(assigned_block)
            if block_alloc is not None:
                boff = block_alloc.offset
                # Clamp the bridge coord into the block's bounding region
                # but keep it at the edge (don't move it inside).
                # The bridge coord from the layout is already correct —
                # just apply the block offset if the raw coord is relative.
                if coord == (0,) or coord == (0, 0):
                    # Fallback: place at the block boundary
                    x = int(boff[0])
                    y = int(boff[1])

            ion = QubitIon("#e8927c", label="A")  # Ancilla colour
            ion.set(gi + 1, x, y)

            # Fix 6: Tag bridge ions with their preferred grid edge.
            # The preferred edge is the direction from the assigned block
            # toward the adjacent block, so the bridge ion is placed near
            # the boundary where it will interact during merge phases.
            _adjacent_block = None
            if gi in _bridge_connected:
                for cb in _bridge_connected[gi]:
                    if cb != assigned_block and cb in block_names_set:
                        _adjacent_block = cb
                        break
            if _adjacent_block is not None:
                _adj_alloc = qubit_allocation.blocks.get(_adjacent_block)
                _own_alloc = qubit_allocation.blocks.get(assigned_block)
                if _adj_alloc is not None and _own_alloc is not None:
                    _own_off = _own_alloc.offset
                    _adj_off = _adj_alloc.offset
                    dx = _adj_off[0] - _own_off[0]
                    dy = _adj_off[1] - _own_off[1]
                    if abs(dx) >= abs(dy):
                        ion._preferred_edge = "right" if dx > 0 else "left"
                    else:
                        # stim y-axis: negative dy = below → high grid row
                        ion._preferred_edge = "bottom" if dy < 0 else "top"

            measurement_ions_per_block.setdefault(assigned_block, []).append(ion)

    return measurement_ions_per_block, data_ions_per_block


# =====================================================================
# Integration Layer: Full Experiment Orchestrator
# =====================================================================

def route_full_experiment(
    qec_meta: "QECMetadata",
    gadget: "Gadget",
    qubit_allocation: "QubitAllocation",
    k: int,
    subgridsize: Tuple[int, int, int] = (6, 4, 1),
    base_pmax_in: int = 1,
    lookahead: int = 4,
    max_inner_workers: int | None = None,
    stop_event: Any = None,
    progress_callback: Any = None,
    measurement_ions_per_block: Optional[Dict[str, List[Any]]] = None,
    data_ions_per_block: Optional[Dict[str, List[Any]]] = None,
    cache_ec_rounds: bool = True,
) -> FullExperimentResult:
    """End-to-end phase-aware routing for a fault-tolerant gadget experiment.

    Orchestrates Steps 2–7 from GADGET_COMPILATION_SPEC.md: for each
    temporal phase, routes on the appropriate (sub-)grid using the
    existing building blocks, stitching results together with
    BT-embedded transitions.

    This function replaces the ``_route_and_simulate()`` call inside
    ``run_single_gadget_config()``.  It calls ``_patch_and_route()``
    directly (Level 1 interface) and computes timing analytically,
    bypassing the ``TrappedIonCompiler`` stack entirely.

    Parameters
    ----------
    qec_meta : QECMetadata
        Rich metadata from ``FaultTolerantGadgetExperiment``.
    gadget : Gadget
        The gadget object.
    qubit_allocation : QubitAllocation
        Full qubit allocation from the experiment.
    k : int
        Trap capacity.
    subgridsize : Tuple[int, int, int]
        ``(width, height, increment)`` for Level 2 patch decomposition.
    base_pmax_in : int
        Base pass horizon for SAT solver.
    lookahead : int
        SAT solver lookahead window.
    max_inner_workers : int, optional
        Max parallel SAT workers.
    stop_event : Event, optional
        Cancellation signal.
    progress_callback : callable, optional
        Progress reporting callback.
    measurement_ions_per_block : Dict[str, List], optional
        If provided, used for ``map_qubits_per_block``.
    data_ions_per_block : Dict[str, List], optional
        If provided, used for ``map_qubits_per_block``.

    Returns
    -------
    FullExperimentResult
        Routing results, schedules, and timing for every phase.
    """
    import logging
    logger = logging.getLogger(__name__)

    # === SETUP: Partition grid into per-block sub-grids (Step 3) ===
    sub_grids = partition_grid_for_blocks(qec_meta, qubit_allocation, k)

    # Create ions from allocation if not provided externally
    if measurement_ions_per_block is None or data_ions_per_block is None:
        measurement_ions_per_block, data_ions_per_block = (
            _create_ions_from_allocation(qec_meta, qubit_allocation)
        )

    # Map qubits per block (Step 3b: initial placement)
    map_qubits_per_block(sub_grids, measurement_ions_per_block, data_ions_per_block, k)

    # Build unified qubit → ion mapping.
    # Ion indices are 1-based (global_idx + 1) to avoid collision
    # with the 0 = empty sentinel in layout arrays.
    qubit_to_ion: Dict[int, int] = {}
    for ba in qec_meta.block_allocations:
        for q in (
            list(ba.data_qubits)
            + list(ba.x_ancilla_qubits)
            + list(ba.z_ancilla_qubits)
        ):
            qubit_to_ion[q] = q + 1

    # Include bridge ancilla qubits (CSS Surgery merge phases use these)
    if qubit_allocation is not None and hasattr(qubit_allocation, 'bridge_ancillas'):
        for bridge_info in (qubit_allocation.bridge_ancillas or []):
            global_idx = bridge_info[0]
            if global_idx not in qubit_to_ion:
                qubit_to_ion[global_idx] = global_idx + 1

    # Decompose into routing plans (Step A1)
    plans = decompose_into_phases(
        qec_meta, gadget, qubit_allocation, sub_grids, qubit_to_ion, k
    )

    logger.info(
        "route_full_experiment: %d phases decomposed, %d blocks, k=%d",
        len(plans), len(sub_grids), k,
    )

    # === BUILD global initial layout from per-block sub_grids ===
    n_rows = max((sg.n_rows for sg in sub_grids.values()), default=0)
    n_cols_total = sum(sg.n_cols for sg in sub_grids.values())
    n_cols_ion = n_cols_total * k

    initial_layout = np.zeros((n_rows, n_cols_ion), dtype=int)
    for name, sg in sub_grids.items():
        if sg.initial_layout is not None:
            r0, c0, r1, c1 = sg.grid_region
            c0i, c1i = c0 * k, c1 * k
            rows = min(r1 - r0, sg.initial_layout.shape[0])
            cols = min(c1i - c0i, sg.initial_layout.shape[1])
            initial_layout[r0:r0 + rows, c0i:c0i + cols] = (
                sg.initial_layout[:rows, :cols]
            )

    # Collect active ions from all blocks
    active_ions: List[int] = []
    for sg in sub_grids.values():
        active_ions.extend(sg.ion_indices)

    # === ROUTE via unified core ===
    all_steps, final_layout = route_full_experiment_as_steps(
        initial_layout=initial_layout,
        n=n_rows,
        m=n_cols_total,
        k=k,
        active_ions=active_ions,
        plans=plans,
        block_sub_grids=sub_grids,
        subgridsize=subgridsize,
        base_pmax_in=base_pmax_in,
        lookahead=lookahead,
        max_inner_workers=max_inner_workers,
        stop_event=stop_event,
        cache_ec_rounds=cache_ec_rounds,
        progress_callback=progress_callback,
    )

    # === GROUP steps by phase into PhaseResult objects ===
    phase_results: List[PhaseResult] = []
    all_schedules: List[List[Dict[str, Any]]] = []
    cached_count = 0
    ms_rounds_routed = 0
    ms_rounds_replayed = 0
    step_cursor = 0
    round_cursor = 0

    for plan in plans:
        n_pairs = len(plan.ms_pairs_per_round)
        if n_pairs <= 0:
            continue

        phase_end = round_cursor + n_pairs - 1
        phase_steps = []
        while (
            step_cursor < len(all_steps)
            and all_steps[step_cursor].ms_round_index <= phase_end
        ):
            phase_steps.append(all_steps[step_cursor])
            step_cursor += 1
        round_cursor += n_pairs

        # Extract schedules and cache status from steps
        phase_schedule = [s.schedule for s in phase_steps if s.schedule]
        from_cache = (
            all(s.from_cache for s in phase_steps)
            if phase_steps
            else False
        )

        # Compute per-phase timing
        ph_exec, ph_reconfig = 0.0, 0.0
        if phase_schedule:
            ph_exec, ph_reconfig, _ = compute_schedule_timing(
                phase_schedule,
                n_rows=n_rows,
                n_cols=n_cols_ion,
                k=k,
                phase_type=plan.phase_type,
                num_stabilizer_rounds=(
                    plan.num_rounds if plan.phase_type == "ec" else 0
                ),
            )

        if from_cache:
            cached_count += 1
            ms_rounds_replayed += n_pairs
        else:
            ms_rounds_routed += n_pairs

        pr = PhaseResult(
            phase_index=plan.phase_index,
            phase_type=plan.phase_type,
            layouts=[s.layout_after for s in phase_steps],
            schedule=phase_schedule,
            exec_time=ph_exec,
            reconfig_time=ph_reconfig,
            from_cache=from_cache,
        )
        phase_results.append(pr)
        all_schedules.extend(phase_schedule)

    # === TIMING ===
    # Global timing: sum per-phase results for accuracy (each phase
    # has its own gate-level overhead), fall back to aggregate if no
    # phase results available.
    if phase_results:
        total_exec = sum(pr.exec_time for pr in phase_results)
        total_reconfig = sum(pr.reconfig_time for pr in phase_results)
        _, _, per_ion_heat = compute_schedule_timing(
            all_schedules, n_rows=n_rows, n_cols=n_cols_ion, k=k,
        )
    else:
        total_exec, total_reconfig, per_ion_heat = compute_schedule_timing(
            all_schedules, n_rows=n_rows, n_cols=n_cols_ion, k=k,
        )

    logger.info(
        "route_full_experiment: total_exec=%.6f s, total_reconfig=%.6f s, "
        "cached=%d/%d phases, routed=%d rounds, replayed=%d rounds",
        total_exec, total_reconfig, cached_count, len(phase_results),
        ms_rounds_routed, ms_rounds_replayed,
    )

    return FullExperimentResult(
        phase_results=phase_results,
        total_schedule=all_schedules,
        total_exec_time=total_exec,
        total_reconfig_time=total_reconfig,
        sub_grids=sub_grids,
        per_ion_heating=per_ion_heat,
        cached_phases=cached_count,
        total_phases=len(phase_results),
        ms_rounds_routed=ms_rounds_routed,
        ms_rounds_replayed=ms_rounds_replayed,
    )


# =====================================================================
# Unified Routing Core: route_full_experiment_as_steps
# =====================================================================

def _build_plans_from_compiler_pairs(
    phases: List[Any],
    parallelPairs: List[List[Tuple[int, int]]],
    phase_pair_counts: List[int],
    block_sub_grids: Dict[str, "BlockSubGrid"],
) -> List["PhaseRoutingPlan"]:
    """Build :class:`PhaseRoutingPlan` from compiler-extracted parallelPairs.

    Called by ``ionRoutingGadgetArch`` to map the compiler's authoritative
    ``parallelPairs`` (from ``toMoveOps``) onto the phase structure from
    ``QECMetadata.phases``.  The *phase_pair_counts* list (from epoch
    analysis) determines how many ``parallelPairs`` entries each phase
    consumes.

    Parameters
    ----------
    phases : list
        Filtered phase objects from ``QECMetadata.phases`` (init/measure
        excluded).
    parallelPairs : list
        All MS-round pair lists from the compiler.
    phase_pair_counts : list of int
        Number of ``parallelPairs`` entries per phase (same length as
        *phases*).
    block_sub_grids : dict
        Per-block sub-grid allocations.

    Returns
    -------
    List[PhaseRoutingPlan]
    """
    all_block_names = list(block_sub_grids.keys())
    plans: List[PhaseRoutingPlan] = []
    pair_cursor = 0

    for i, (phase, n_pairs) in enumerate(zip(phases, phase_pair_counts)):
        phase_type = getattr(phase, 'phase_type', '') or ''
        active_blocks = (
            getattr(phase, 'active_blocks', None)
            if getattr(phase, 'active_blocks', None) is not None
            else list(all_block_names)
        )
        idle_blocks = [b for b in all_block_names if b not in active_blocks]

        _is_ec = _is_ec_phase_type(phase_type)

        if n_pairs <= 0:
            plans.append(PhaseRoutingPlan(
                phase_type='ec' if _is_ec else phase_type,
                phase_index=i,
                interacting_blocks=list(active_blocks),
                idle_blocks=idle_blocks,
                all_blocks=list(all_block_names),
                ms_pairs_per_round=[],
                num_rounds=getattr(phase, 'num_rounds', 0) or 0,
            ))
            continue

        phase_pairs = parallelPairs[pair_cursor:pair_cursor + n_pairs]
        pair_cursor += n_pairs

        # ----------------------------------------------------------
        # Block-aware pair filtering for EC phases.
        #
        # After Fix10 expansion the blind cursor slice may assign
        # parallelPairs entries whose ions belong to a block that is
        # *not* active in this phase.  For EC phases we filter out
        # such "foreign" pairs so that the routing plan only contains
        # pairs involving ions from the phase's active blocks.
        # ----------------------------------------------------------
        if _is_ec and block_sub_grids and active_blocks:
            _active_ions: set = set()
            for _bn in active_blocks:
                _bsg = block_sub_grids.get(_bn)
                if _bsg is not None:
                    if _bsg.ion_indices:
                        _active_ions.update(_bsg.ion_indices)
                    elif _bsg.qubit_to_ion:
                        _active_ions.update(_bsg.qubit_to_ion.values())
            if _active_ions:
                _filtered: List[List[Tuple[int, int]]] = []
                _dropped = 0
                for _rp in phase_pairs:
                    _kept = [
                        (a, d) for a, d in _rp
                        if a in _active_ions and d in _active_ions
                    ]
                    _dropped += len(_rp) - len(_kept)
                    if _kept:
                        _filtered.append(_kept)
                if _dropped > 0:
                    logger.info(
                        "[GadgetRouting] Phase %d (%s): filtered %d "
                        "foreign pairs from %d rounds → %d rounds "
                        "(active_blocks=%s)",
                        i, phase_type, _dropped,
                        len(phase_pairs), len(_filtered),
                        active_blocks,
                    )
                    phase_pairs = _filtered

        round_sig = getattr(phase, 'round_signature', None)

        plans.append(PhaseRoutingPlan(
            phase_type='ec' if _is_ec else phase_type,
            phase_index=i,
            interacting_blocks=list(active_blocks),
            idle_blocks=idle_blocks,
            all_blocks=list(all_block_names),
            ms_pairs_per_round=phase_pairs,
            num_rounds=getattr(phase, 'num_rounds', 1) or 1,
            round_signature=round_sig,
        ))

    return plans



# =====================================================================
# Merge adjacent parallelPairs rounds with disjoint ion sets
# =====================================================================

def _merge_phase_pairs(
    phase_pairs: List[List[Tuple[int, int]]],
) -> List[List[Tuple[int, int]]]:
    """Merge adjacent routing rounds whose ion sets are completely disjoint.

    When EC rounds are emitted sequentially per block in the stim
    circuit, each block's CX becomes a separate ``parallelPairs`` entry
    with 1-2 pairs.  Cross-block CX that operate on entirely different
    ions can physically execute in the same routing round.  Merging them
    reduces the number of reconfiguration cycles and improves
    parallelism.

    The merge is conservative: two rounds are combined only when *every*
    ion in the first round is absent from the second round (and vice
    versa).  This guarantees no data-dependency violations.
    """
    if len(phase_pairs) <= 1:
        return phase_pairs
    merged: List[List[Tuple[int, int]]] = []
    for pairs in phase_pairs:
        if not pairs:
            continue
        if not merged:
            merged.append(list(pairs))
            continue
        # Collect ion indices in last merged group
        last_ions: set = set()
        for a, d in merged[-1]:
            last_ions.add(a)
            last_ions.add(d)
        # Collect ion indices in current group
        curr_ions: set = set()
        for a, d in pairs:
            curr_ions.add(a)
            curr_ions.add(d)
        if last_ions.isdisjoint(curr_ions):
            merged[-1].extend(pairs)
        else:
            merged.append(list(pairs))
    return merged


# =====================================================================
# Fix 7: Split rounds with shared-ion pairs into disjoint sub-rounds
# =====================================================================

def _split_shared_ion_rounds(
    phase_pairs: List[List[Tuple[int, int]]],
) -> List[List[Tuple[int, int]]]:
    """Split routing rounds where an ion appears in multiple pairs.

    The SAT solver requires **all** pairs within a round to be co-located
    simultaneously (same row + same trap block).  When pairs share a
    common ion (e.g. bridge ancilla in CSS Surgery), simultaneous
    co-location is physically impossible, and the solver returns UNSAT.

    This function detects such rounds and splits them into sub-rounds
    where every pair's ions are disjoint.  The greedy packing maximises
    parallelism within each sub-round (different bridge ancillas whose
    data qubits don't overlap can still share a sub-round).

    Rounds that already have fully disjoint pairs are returned as-is.
    """
    result: List[List[Tuple[int, int]]] = []
    for pairs in phase_pairs:
        if not pairs:
            result.append(pairs)
            continue
        # Quick check: any ion appears more than once?
        ion_counts: Dict[int, int] = {}
        for a, b in pairs:
            ion_counts[a] = ion_counts.get(a, 0) + 1
            ion_counts[b] = ion_counts.get(b, 0) + 1
        has_shared = any(c > 1 for c in ion_counts.values())
        if not has_shared:
            result.append(pairs)
            continue
        # Greedy packing into disjoint sub-rounds
        remaining = list(pairs)
        while remaining:
            sub_round: List[Tuple[int, int]] = []
            used_ions: set = set()
            leftover: List[Tuple[int, int]] = []
            for pair in remaining:
                a, b = pair
                if a not in used_ions and b not in used_ions:
                    sub_round.append(pair)
                    used_ions.add(a)
                    used_ions.add(b)
                else:
                    leftover.append(pair)
            result.append(sub_round)
            remaining = leftover
    return result


# =====================================================================
# Fix 4 Helper: Reconstruct EC target layout from per-block snapshots
# =====================================================================

def _reconstruct_ec_target(
    ec_initial_layouts: Dict[str, np.ndarray],
    global_layout: np.ndarray,
    block_sub_grids: Dict[str, "BlockSubGrid"],
    k: int,
    ion_to_block: Optional[Dict[int, str]] = None,
) -> np.ndarray:
    """Build a full-grid target from per-block EC initial layouts.

    When *ion_to_block* is supplied, only **active** ions (those that
    carry a qubit) are pinned to their EC-initial positions.  Spectator
    ions stay at their **current** positions in *global_layout*, which
    drastically reduces the permutation the SAT solver must achieve.
    If a spectator currently occupies a cell needed by an active ion,
    the spectator is relocated to the nearest available empty cell.

    Without *ion_to_block* this falls back to the legacy behaviour of
    overwriting entire block regions.
    """
    if ion_to_block is None:
        # Legacy path: overwrite entire block regions (moves spectators)
        target = global_layout.copy()
        for bname, sg in block_sub_grids.items():
            if bname in ec_initial_layouts:
                r0, c0, r1, c1 = sg.grid_region
                c0i, c1i = c0 * k, c1 * k
                ec_lay = ec_initial_layouts[bname]
                rows = min(r1 - r0, ec_lay.shape[0])
                cols = min(c1i - c0i, ec_lay.shape[1])
                target[r0:r0 + rows, c0i:c0i + cols] = ec_lay[:rows, :cols]
        return target

    # ── Smart path: only pin active ions ──
    # 1. Collect where each active ion must go (from EC-initial layouts)
    active_targets: Dict[int, Tuple[int, int]] = {}
    for bname, sg in block_sub_grids.items():
        if bname not in ec_initial_layouts:
            continue
        r0, c0, r1, c1 = sg.grid_region
        c0i = c0 * k
        ec_lay = ec_initial_layouts[bname]
        rows = min(r1 - r0, ec_lay.shape[0])
        cols = min((c1 - c0) * k, ec_lay.shape[1])
        for r in range(rows):
            for c in range(cols):
                ion_idx = int(ec_lay[r, c])
                if ion_idx != 0 and ion_idx in ion_to_block:
                    active_targets[ion_idx] = (r0 + r, c0i + c)

    # 2. Start from the *current* layout (preserves spectator positions)
    target = global_layout.copy()

    # 3. Remove all active ions from their current positions
    for r in range(target.shape[0]):
        for c in range(target.shape[1]):
            if int(target[r, c]) in active_targets:
                target[r, c] = 0

    # 4. Place active ions at their EC-initial positions.
    #    If a spectator sits there, displace it.
    #    Track which block the displaced spectator belongs to so we can
    #    relocate it within its home block (FM4 fix).
    displaced_with_home: list = []  # (ion_idx, home_block_name_or_None)
    for ion_idx, (tr, tc) in active_targets.items():
        occupant = int(target[tr, tc])
        if occupant != 0:
            home_block = ion_to_block.get(occupant)  # may be None for spectators
            displaced_with_home.append((occupant, home_block))
            target[tr, tc] = 0
        target[tr, tc] = ion_idx

    # 5. Relocate displaced spectators to empty cells IN THEIR HOME BLOCK.
    #    This prevents spectators from being placed in foreign block
    #    regions, which would make the target layout itself block-impure
    #    and cause all downstream route-back to fail (FM4 root cause).
    # Pre-build block → grid region lookup (ion-column coordinates)
    _block_regions: Dict[str, Tuple[int, int, int, int]] = {}
    for _bname, _sg in block_sub_grids.items():
        _r0, _c0, _r1, _c1 = _sg.grid_region
        _block_regions[_bname] = (_r0, _c0 * k, _r1, _c1 * k)

    for spec_ion, home_block in displaced_with_home:
        placed = False
        # Try home block first (if the spectator is mapped to a block)
        if home_block and home_block in _block_regions:
            hr0, hc0, hr1, hc1 = _block_regions[home_block]
            for r in range(hr0, hr1):
                for c in range(hc0, hc1):
                    if target[r, c] == 0:
                        target[r, c] = spec_ion
                        placed = True
                        break
                if placed:
                    break
        if not placed:
            # Unmapped spectator or no space in home block — any empty cell
            for r in range(target.shape[0]):
                for c in range(target.shape[1]):
                    if target[r, c] == 0:
                        target[r, c] = spec_ion
                        placed = True
                        break
                if placed:
                    break
        if not placed:
            raise ValueError(
                f"_reconstruct_ec_target: no empty cell for displaced "
                f"spectator ion {spec_ion} (home_block={home_block})"
            )

    return target


# =====================================================================
# Fix 4 Helper: SAT-based transition reconfig between layouts
# =====================================================================

def _compute_transition_reconfig_steps(
    current_layout: np.ndarray,
    target_layout: np.ndarray,
    wiseArch: "QCCDWiseArch",
    block_sub_grids: Dict[str, "BlockSubGrid"],
    ion_to_block: Dict[int, str],
    k: int,
    *,
    subgridsize: Tuple[int, int, int],
    base_pmax_in: int = 1,
    max_inner_workers: int | None = None,
    stop_event: Any = None,
    progress_callback: Any = None,
    allow_heuristic_fallback: bool = False,
    use_heuristic_directly: bool = False,
    max_sat_time: Optional[float] = None,
    max_rc2_time: Optional[float] = None,
    solver_params: Optional[Any] = None,
) -> List:
    """Compute SAT-based transition reconfig from current to target layout.

    Delegates to ``_rebuild_schedule_for_layout`` which uses the same
    patch-and-route machinery (checkerboard offset tilings, multi-cycle
    convergence with BT pins) but is driven by BT targets rather than
    MS-gate pairs.

    For per-block spatial slicing (when all ions are in their correct
    block regions), a sub-``QCCDWiseArch`` is built per block and
    ``_rebuild_schedule_for_layout`` is called on each block independently,
    then merged via ``_merge_block_routing_steps``.

    Returns a list of RoutingStep objects that transition the layout.
    """
    from ..compiler.qccd_WISE_ion_route import (
        _rebuild_schedule_for_layout,
        _merge_block_routing_steps,
        RoutingStep,
    )
    from ..compiler.qccd_SAT_WISE_odd_even_sorter import (
        NoFeasibleLayoutError,
    )
    from .qccd_nodes import QCCDWiseArch as _WA

    import logging
    _logger = logging.getLogger("wise.qccd.gadget_routing")

    if np.array_equal(current_layout, target_layout):
        return []

    n_rows, n_cols = current_layout.shape

    # ── Fast path: skip SAT entirely when heuristic is requested ──
    # The plan phase (MS routing) is already complete.  The heuristic
    # (odd-even transposition sort) always produces the exact target
    # for any valid permutation — no need to invoke SAT.
    if use_heuristic_directly:
        _mismatch = int(np.count_nonzero(current_layout != target_layout))
        _logger.info(
            "[TransitionReconfig] skipping SAT — using heuristic directly "
            "(%d non-matching cells)",
            _mismatch,
        )
        return [RoutingStep(
            layout_after=np.array(target_layout, copy=True),
            schedule=None,
            solved_pairs=[],
            ms_round_index=-1,
            from_cache=False,
            tiling_meta=(0, 0),
            can_merge_with_next=False,
            is_initial_placement=False,
            is_layout_transition=True,
            reconfig_context="cache_replay",
            layout_before=np.array(current_layout, copy=True),
        )]

    def _snapshots_to_steps(snaps, step_offset=0):
        """Convert _rebuild_schedule_for_layout snapshots to RoutingStep list."""
        steps = []
        _prev_lay = np.array(current_layout, copy=True)
        for i, (layout, schedule, _) in enumerate(snaps):
            steps.append(RoutingStep(
                layout_after=np.array(layout, copy=True),
                schedule=schedule,
                solved_pairs=[],
                ms_round_index=-1,
                from_cache=False,
                tiling_meta=(step_offset + i, 0),
                can_merge_with_next=False,
                is_initial_placement=False,
                is_layout_transition=True,
                reconfig_context="cache_replay",
                layout_before=_prev_lay,
            ))
            _prev_lay = np.array(layout, copy=True)
        return steps

    def _consolidate_transition_steps(steps):
        """Merge multiple transition RoutingSteps into a single step.

        Concatenates all SAT schedules sequentially and uses the last
        step's ``layout_after``.  This reduces multiple global
        reconfigurations into one, which is valid because each
        snapshot's schedule transforms the previous layout to the next
        and concatenation preserves this sequential property.

        If any step has ``schedule=None`` (heuristic fallback from
        ``_rebuild_schedule_for_layout``), the consolidated result also
        uses ``schedule=None``.  This causes ``physicalOperation`` to
        fall through to the deterministic Phase B/C/D odd-even
        transposition sort on the full grid, which always succeeds.
        """
        if len(steps) <= 1:
            return steps
        # If any step uses the heuristic fallback (schedule=None), the
        # earlier SAT schedules only got us partway.  The heuristic will
        # sort from the current layout all the way to the final target
        # in one shot, so we consolidate to schedule=None.
        has_heuristic_fallback = any(s.schedule is None for s in steps)
        if has_heuristic_fallback:
            return [RoutingStep(
                layout_after=np.array(steps[-1].layout_after, copy=True),
                schedule=None,
                solved_pairs=[],
                ms_round_index=-1,
                from_cache=False,
                tiling_meta=(0, 0),
                can_merge_with_next=False,
                is_initial_placement=False,
                is_layout_transition=True,
                reconfig_context="cache_replay",
                layout_before=(
                    np.array(steps[0].layout_before, copy=True)
                    if steps[0].layout_before is not None else None
                ),
            )]
        merged_schedule = []
        for s in steps:
            if s.schedule:
                merged_schedule.extend(s.schedule)
        return [RoutingStep(
            layout_after=np.array(steps[-1].layout_after, copy=True),
            schedule=merged_schedule if merged_schedule else None,
            solved_pairs=[],
            ms_round_index=-1,
            from_cache=False,
            tiling_meta=(0, 0),
            can_merge_with_next=False,
            is_initial_placement=False,
            is_layout_transition=True,
            reconfig_context="cache_replay",
            layout_before=(
                np.array(steps[0].layout_before, copy=True)
                if steps[0].layout_before is not None else None
            ),
        )]

    # --- Check block purity ---
    # Only consider ions that are actually mapped to a block (i.e.
    # carry a qubit).  Unmapped ions are physical trap slots with
    # no qubit assignment (common when k > 1) and must be ignored.
    blocks_pure = True
    for bname, sg in block_sub_grids.items():
        r0, c0, r1, c1 = sg.grid_region
        c0i, c1i = c0 * k, c1 * k
        block_slice = current_layout[r0:r1, c0i:c1i]
        for r in range(block_slice.shape[0]):
            for c in range(block_slice.shape[1]):
                ion_idx = int(block_slice[r, c])
                if ion_idx != 0 and ion_idx in ion_to_block:
                    owner = ion_to_block[ion_idx]
                    if owner != bname:
                        blocks_pure = False
                        break
            if not blocks_pure:
                break
        if not blocks_pure:
            break

    # --- Always use full-grid routing for transitions ---
    # Route-back should be batched across ALL blocks simultaneously
    # on the full grid to minimise cost and coordinate ion movements
    # at block boundaries. Per-block slicing is not used for route-back.
    _logger.info(
        "[TransitionReconfig] using full-grid batched routing across "
        "all blocks (%d blocks, blocks_pure=%s)",
        len(block_sub_grids), blocks_pure,
    )

    # FM6: SAT-first transition reconfig.  Heuristic fallback is controlled
    # by the caller: cache-replay contexts pass allow_heuristic_fallback=True
    # (the heuristic always converges and is acceptable per P2); non-cache
    # contexts (e.g. return-reconfig) pass False and rely on SAT escalation.
    snaps = _rebuild_schedule_for_layout(
        current_layout.copy(), wiseArch, target_layout,
        subgridsize=subgridsize,
        base_pmax_in=base_pmax_in,
        stop_event=stop_event,
        max_inner_workers=max_inner_workers,
        progress_callback=progress_callback,
        allow_heuristic_fallback=allow_heuristic_fallback,
        max_sat_time=max_sat_time,
        max_rc2_time=max_rc2_time,
        solver_params=solver_params,
    )
    # Consolidate multiple transition steps into one reconfiguration
    return _consolidate_transition_steps(_snapshots_to_steps(snaps))


def _is_ec_phase_type(phase_type: str) -> bool:
    """Return True if *phase_type* represents an EC / stabilizer phase."""
    return (
        phase_type in ('ec', 'stabilizer_round', 'final_round')
        or phase_type.startswith('stabilizer_round')
    )


def _build_gadget_exit_bt(
    sgs_with_offsets: List[Tuple[str, int, int]],
    phase_pairs: list,
    ec_initial_layouts: Dict[str, np.ndarray],
) -> Tuple[list, Optional[list]]:
    """Build exit BT constraints and solve-array for gadget routing.

    Parameters
    ----------
    sgs_with_offsets : list of ``(block_name, r0_offset, c0_offset)``
    phase_pairs : MS pairs for this phase.
    ec_initial_layouts : per-block EC starting layouts.

    Returns ``(p_arr_for_solve, bts)`` — *bts* is ``None`` when no
    BT is needed.
    """
    if not ec_initial_layouts:
        return list(phase_pairs), None
    exit_bt: Dict[int, Tuple[int, int]] = {}
    for bname, r0_off, c0_off in sgs_with_offsets:
        if bname not in ec_initial_layouts:
            continue
        ec_lay = ec_initial_layouts[bname]
        for r in range(ec_lay.shape[0]):
            for c in range(ec_lay.shape[1]):
                ion_idx = int(ec_lay[r, c])
                if ion_idx != 0:
                    exit_bt[ion_idx] = (r + r0_off, c + c0_off)
    if not exit_bt:
        return list(phase_pairs), None
    p_arr = list(phase_pairs) + [[]]
    bts: list = [dict() for _ in range(len(p_arr))]
    bts[-1] = {(0, 0): exit_bt}
    return p_arr, bts


def _preprocess_gadget_pairs(
    phase_pairs: List[List[Tuple[int, int]]],
    ion_to_block: Dict[int, str],
    phase_idx: int,
    logger,
) -> Tuple[List[List[Tuple[int, int]]], int]:
    """Split and decompose gadget MS rounds for SAT-friendliness.

    1. **Fix 7** — Split rounds where pairs share an ion (e.g. bridge
       ancillas in CSS Surgery) into disjoint sub-rounds.
    2. **Bug 3/4** — Decompose multi-block rounds into per-block
       sub-rounds so the SAT solver handles smaller problems.

    Returns ``(processed_pairs, n_pairs)``.
    """
    from collections import defaultdict as _ddict

    # Fix 7: split shared-ion rounds
    _orig_len = len(phase_pairs)
    phase_pairs = _split_shared_ion_rounds(phase_pairs)
    if len(phase_pairs) != _orig_len:
        logger.info(
            "[PhaseSteps] gadget phase %d: split shared-ion "
            "rounds: %d → %d sub-rounds",
            phase_idx, _orig_len, len(phase_pairs),
        )

    n_pairs = len(phase_pairs)

    # Bug 3/4: decompose multi-block intra-block rounds
    if ion_to_block:
        _decomposed: List[List[Tuple[int, int]]] = []
        for _rnd in phase_pairs:
            _blk_pairs: Dict[str, List[Tuple[int, int]]] = _ddict(list)
            _cross: List[Tuple[int, int]] = []
            for _p in _rnd:
                _ba = ion_to_block.get(_p[0])
                _bb = ion_to_block.get(_p[1])
                if _ba is not None and _bb is not None and _ba == _bb:
                    _blk_pairs[_ba].append(_p)
                else:
                    _cross.append(_p)
            if _cross:
                _decomposed.append(_cross)
            for _bps in _blk_pairs.values():
                if _bps:
                    _decomposed.append(_bps)
        if len(_decomposed) != len(phase_pairs):
            logger.info(
                "[PhaseSteps] gadget phase %d: decomposed "
                "multi-block rounds: %d → %d sub-rounds",
                phase_idx, len(phase_pairs), len(_decomposed),
            )
            phase_pairs = _decomposed
            n_pairs = len(phase_pairs)

    return phase_pairs, n_pairs


def route_full_experiment_as_steps(
    initial_layout: np.ndarray,
    n: int,
    m: int,
    k: int,
    active_ions: List[int],
    plans: List[PhaseRoutingPlan],
    block_sub_grids: Dict[str, BlockSubGrid],
    *,
    subgridsize: Optional[Tuple[int, int, int]] = None,
    base_pmax_in: int = 1,
    lookahead: int = 4,
    max_inner_workers: int | None = None,
    stop_event: Any = None,
    cx_per_ec_round: Optional[int] = None,
    ms_rounds_per_ec_round: Optional[int] = None,
    cache_ec_rounds: bool = True,
    replay_level: int = 1,
    block_level_slicing: bool = True,
    heuristic_cache_replay: bool = False,
    heuristic_route_back: bool = False,
    heuristic_fallback_for_noncache: bool = False,
    progress_callback: Any = None,
    max_sat_time: Optional[float] = None,
    max_rc2_time: Optional[float] = None,
    solver_params: Any = None,
) -> Tuple[List, np.ndarray]:
    """Phase-aware routing returning ``RoutingStep`` objects.

    Unified routing core used by both the compiler path
    (``ionRoutingGadgetArch``) and the timing-only path
    (``route_full_experiment``).  Takes the *best* of both original
    implementations:

    - **From ``gadget_routing``**: metadata-based phase decomposition,
      ``map_qubits_per_block`` (hill-climb initial layout), exit BT on
      gadget phases, ``build_merged_layout`` for Level 1 slicing.
    - **From ``ionRoutingGadgetArch``**: direct ``RoutingStep``
      production, ``_merge_block_routing_steps``, per-stabilizer-round
      splitting with block decomposition.

    EC phases
    ~~~~~~~~~
    Each EC phase is split into per-stabilizer-round chunks (when
    *cx_per_ec_round* is provided).  Within each chunk, blocks are
    routed independently on disjoint sub-grids and merged via
    ``_merge_block_routing_steps``.  Ion-return BT ensures ions return
    to starting positions.  Round-signature caching avoids re-solving
    identical phases.

    Gadget phases
    ~~~~~~~~~~~~~
    Level 1 spatial slicing merges only the interacting blocks into a
    compact sub-grid.  Exit BT pins ions to their most-recent EC
    positions so the next EC phase starts from a known layout.
    Falls back gracefully if BT causes UNSAT.

    Parameters
    ----------
    initial_layout : np.ndarray
        Global ``n × (m * k)`` ion arrangement array.
    n, m, k : int
        Grid geometry: rows, traps-per-row, ions-per-trap.
    active_ions : list of int
        Non-spectator ion indices.
    plans : list of PhaseRoutingPlan
        Phase decomposition with MS pairs per phase.
    block_sub_grids : dict of str to BlockSubGrid
        Per-block sub-grid allocations.
    subgridsize : tuple
        ``(width, height, increment)`` for patch decomposition.
    base_pmax_in : int
        Base pass horizon for the SAT solver.
    lookahead : int
        SAT solver lookahead window.
    max_inner_workers : int, optional
        Max parallel SAT workers.
    stop_event : optional
        Cancellation signal.
    cx_per_ec_round : int, optional
        CX pairs per single EC stabilizer round.  Splits EC phases
        into per-stabilizer-round chunks for faster SAT solving.
    cache_ec_rounds : bool
        When ``True`` (default), identical EC phases (same
        ``round_signature``) are cached and replayed instead of
        re-solved.  Set ``False`` to force fresh SAT routing on
        every EC phase.

    Heuristic fallback policy — DO NOT REVERT
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    The three heuristic flags (``heuristic_cache_replay``,
    ``heuristic_route_back``, ``heuristic_fallback_for_noncache``)
    control whether the odd-even transposition sort can be used
    instead of SAT for specific contexts:

    - **MS-gate routing**: ALWAYS uses SAT (``allow_heuristic_fallback=False``).
      This is enforced in the ChainVerify loop below, where
      ``_cv_allow_heuristic`` is set to ``False`` when
      ``reconfig_context == "ms_gate"``.  Do not change this.
    - **Return reconfigs**: ``allow_heuristic_fallback=True`` because
      these are pure layout permutations.
    - **Cache replay**: uses ``heuristic_cache_replay`` flag.
    - **Block-merge / transition reconfigs**: uses
      ``heuristic_fallback_for_noncache`` flag.

    If you set ``allow_heuristic_fallback=True`` for MS-gate contexts,
    the SAT solver will be bypassed for MS-gate reconfigs.  This produces
    valid layouts but sub-optimal physical gate schedules.

    Returns
    -------
    Tuple[List[RoutingStep], np.ndarray]
        ``(routing_steps, final_layout)`` where *routing_steps* can
        be passed to ``ionRoutingWISEArch(_precomputed_routing_steps=...)``.
    """
    from ..compiler.qccd_WISE_ion_route import (
        RoutingStep,
        _route_round_sequence,
        _merge_block_routing_steps,
        _remap_schedule_to_global,
        _rebuild_schedule_for_layout,
        _simulate_schedule_replay,
    )
    from ..compiler.routing_config import (
        RoutingProgress,
        STAGE_ROUTING,
        STAGE_RECONFIG,
        STAGE_RECONFIG_PROGRESS,
        STAGE_COMPLETE,
        STAGE_PHASE,
        STAGE_BLOCK,
    )
    from .qccd_nodes import QCCDWiseArch

    # Resolve subgridsize=None → full grid (no patching)
    if subgridsize is None:
        subgridsize = (m * k, n, 1)

    # ── Propagate routing-config timeouts into solver_params ────
    # WISESolverParams.from_grid() doesn't set notebook_sat_timeout
    # or notebook_rc2_timeout.  If the caller set these on the
    # routing config but not on solver_params, copy them across so
    # downstream SAT calls respect user-configured per-call caps.
    import os as _os
    if solver_params is not None:
        if getattr(solver_params, 'notebook_sat_timeout', None) is None:
            _cfg_nst = getattr(max_sat_time, '__self__', None)  # not useful
            # Try to pick up from the keyword args that were forwarded
            # from WISERoutingConfig.  The caller typically passes
            # max_sat_time and max_rc2_time from routing_cfg; notebook
            # timeouts are separate fields on WISERoutingConfig but
            # aren't forwarded to route_full_experiment_as_steps.
            # Best-effort: if max_sat_time looks like a notebook-scale
            # cap (< 600s), use it as the notebook_sat_timeout.
            pass  # handled below via env-var fallback
        _env_ip = int(_os.environ.get('WISE_INPROCESS_LIMIT', '0'))
        if getattr(solver_params, 'inprocess_limit', None) is None and _env_ip:
            solver_params.inprocess_limit = _env_ip
        _env_mac_cap = int(_os.environ.get('WISE_MACOS_THREAD_CAP', '0'))
        if getattr(solver_params, 'macos_thread_cap', None) is None and _env_mac_cap:
            solver_params.macos_thread_cap = _env_mac_cap

    import logging
    logger = logging.getLogger("wise.qccd.gadget_routing")

    all_routing_steps: List[RoutingStep] = []
    current_layout = np.array(initial_layout, copy=True)
    active_ions_set = set(active_ions)
    round_cursor = 0

    # ── Global progress wrapper ──────────────────────────────────
    # _route_round_sequence emits STAGE_ROUTING with per-phase
    # current/total.  The wrapper translates these to global values
    # so the Route bar shows cumulative progress across all phases.
    _total_global_rounds = sum(
        len(p.ms_pairs_per_round)
        for p in plans
        if len(p.ms_pairs_per_round) > 0
    )
    # Estimate reconfig work units (return + transition reconfigs
    # that run *after* all MS rounds finish).  Each phase with MS
    # pairs needs at least one return or transition reconfig.
    _total_reconfig_estimate = sum(
        1 for _p in plans if len(_p.ms_pairs_per_round) > 0
    )
    _total_work_units = _total_global_rounds + _total_reconfig_estimate
    _global_round_offset = 0  # accumulated after each phase
    _route_emit_hwm = 0       # suppress backward Route emissions

    # Reconfig progress counter — tracks transition/return reconfigs
    # across all phases so the user sees deterministic progress after
    # the MS rounds complete.
    _reconfig_count = 0

    def _emit_reconfig_progress(message: str) -> None:
        """Emit a STAGE_RECONFIG_PROGRESS event with advancing counter."""
        nonlocal _reconfig_count
        _reconfig_count += 1
        if progress_callback is None:
            return
        progress_callback(RoutingProgress(
            stage=STAGE_RECONFIG_PROGRESS,
            current=min(_total_global_rounds + _reconfig_count,
                        _total_work_units),
            total=_total_work_units,
            message=message,
        ))

    def _global_progress_wrapper(p: RoutingProgress) -> None:
        """Intercept per-phase routing progress and remap to global."""
        nonlocal _global_round_offset, _route_emit_hwm
        if progress_callback is None:
            return
        if p.stage in (STAGE_ROUTING, STAGE_COMPLETE):
            global_current = _global_round_offset + p.current
            # Suppress backward Route emissions.  Per-block EC routing
            # makes multiple _route_round_sequence calls covering the
            # same MS rounds in different grid regions; without HWM the
            # second block restarts at offset+0 which is < the first
            # block's offset+N.
            if global_current < _route_emit_hwm:
                return
            _route_emit_hwm = global_current
            progress_callback(RoutingProgress(
                stage=p.stage if p.stage != STAGE_COMPLETE else STAGE_ROUTING,
                current=min(global_current, _total_work_units),
                total=_total_work_units,
                gates_remaining=p.gates_remaining,
                elapsed_seconds=p.elapsed_seconds,
                message=f"Phase routing: {global_current}/{_total_work_units}",
                extra=p.extra,
            ))
        elif p.stage in (STAGE_RECONFIG, STAGE_RECONFIG_PROGRESS):
            # Reconfig events: pass through directly
            progress_callback(p)
        else:
            # SAT and Patch stages: pass through unchanged
            progress_callback(p)

    _phase_cb = _global_progress_wrapper if progress_callback is not None else None

    # ── Phase / block progress helpers ───────────────────────────
    _n_phases = len(plans)

    def _emit_phase(phase_idx: int, description: str) -> None:
        """Emit a STAGE_PHASE event with the given description."""
        if progress_callback is None:
            return
        progress_callback(RoutingProgress(
            stage=STAGE_PHASE,
            current=phase_idx,
            total=_n_phases,
            message=description,
        ))

    def _emit_block(block_idx: int, total_blocks: int,
                    block_name: str = "") -> None:
        """Emit a STAGE_BLOCK event for per-block progress."""
        if progress_callback is None:
            return
        progress_callback(RoutingProgress(
            stage=STAGE_BLOCK,
            current=block_idx,
            total=total_blocks,
            message=block_name,
        ))

    # Emit initial Route bar state so the widget shows the total immediately
    if progress_callback is not None and _total_work_units > 0:
        progress_callback(RoutingProgress(
            stage=STAGE_ROUTING,
            current=0,
            total=_total_work_units,
            message=f"Starting: 0/{_total_work_units} steps ({_total_global_rounds} MS + {_total_reconfig_estimate} reconfig)",
        ))

    # Per-block EC layouts for gadget exit BT
    ec_initial_layouts: Dict[str, np.ndarray] = {}

    # EC round-signature cache: maps signature → (starting_layout, steps)
    # so that replays can compute a transition reconfig instead of
    # re-routing every step heuristically.
    ec_cache: Dict[Tuple, Tuple[np.ndarray, List[RoutingStep]]] = {}

    # ── replay_level semantics ───────────────────────────────────
    # replay_level=0 → disable all caching and route-back;
    #                   every round solved fresh with SAT.
    # replay_level≥1 → caching enabled; route-back groups of
    #                   replay_level stabiliser rounds.
    # Override cache_ec_rounds when replay_level=0.
    if replay_level == 0:
        cache_ec_rounds = False

    # Default ms_rounds_per_ec_round from cx_per_ec_round when not
    # supplied.  For gadget experiments they are already equal (both
    # use max(|X|, |Z|)).  For CSS memory cx_per_ec_round is the sum,
    # so a dedicated ms_rounds_per_ec_round from QECMetadata is needed
    # to get the correct interleaved round count.
    if ms_rounds_per_ec_round is None:
        ms_rounds_per_ec_round = cx_per_ec_round  # fallback

    wiseArch = QCCDWiseArch(n=n, m=m, k=k)

    _use_per_block = (
        block_level_slicing
        and block_sub_grids is not None
        and len(block_sub_grids) >= 2
    )

    # Pre-build ion → block mapping
    ion_to_block: Dict[int, str] = {}
    if _use_per_block:
        for bname, sg in block_sub_grids.items():
            for iidx in sg.ion_indices:
                ion_to_block[iidx] = bname

    # Store initial per-block layouts for gadget exit BT
    if _use_per_block:
        for bname, sg in block_sub_grids.items():
            r0, c0, r1, c1 = sg.grid_region
            c0i, c1i = c0 * k, c1 * k
            ec_initial_layouts[bname] = current_layout[r0:r1, c0i:c1i].copy()

    logger.info(
        "[PhaseSteps] routing %d phases, grid=%dx%d, k=%d, blocks=%d",
        len(plans), n, m * k, k, len(block_sub_grids),
    )

    # ── Diagnostic: initial layout purity check ──────────────────
    if ion_to_block and block_sub_grids:
        logger.debug(
            "[BlockPurity-DIAG] grid shape=%s, ion_to_block=%s",
            current_layout.shape,
            {b: sorted(ions) for b, ions in
             {bn: [i for i, bx in ion_to_block.items() if bx == bn]
              for bn in block_sub_grids}.items()},
        )
        for _bname, _sg in block_sub_grids.items():
            _r0, _c0, _r1, _c1 = _sg.grid_region
            _c0i, _c1i = _c0 * k, _c1 * k
            logger.debug(
                "[BlockPurity-DIAG] %s: region=(%d,%d,%d,%d), ion_cols=(%d,%d), "
                "ion_indices=%s, grid_slice=\n%s",
                _bname, _r0, _c0, _r1, _c1, _c0i, _c1i,
                sorted(_sg.ion_indices),
                current_layout[_r0:_r1, _c0i:_c1i],
            )
        _init_misplaced: Dict[str, List[int]] = {}
        for _bname, _sg in block_sub_grids.items():
            _r0, _c0, _r1, _c1 = _sg.grid_region
            _c0i, _c1i = _c0 * k, _c1 * k
            _blk_slice = current_layout[_r0:_r1, _c0i:_c1i]
            for _ri in range(_blk_slice.shape[0]):
                for _ci in range(_blk_slice.shape[1]):
                    _ion = int(_blk_slice[_ri, _ci])
                    if _ion != 0 and _ion in ion_to_block and ion_to_block[_ion] != _bname:
                        _init_misplaced.setdefault(_bname, []).append(_ion)
        if _init_misplaced:
            logger.warning(
                "[BlockPurity] INITIAL LAYOUT IMPURE — misplaced ions: %s. "
                "ion_to_block has %d entries, grid has %d non-zero cells.",
                {b: sorted(ions) for b, ions in _init_misplaced.items()},
                len(ion_to_block),
                int(np.count_nonzero(current_layout)),
            )
            # Log unmapped ions for debugging
            _all_grid_ions = set()
            for _ri in range(current_layout.shape[0]):
                for _ci in range(current_layout.shape[1]):
                    _ion = int(current_layout[_ri, _ci])
                    if _ion != 0:
                        _all_grid_ions.add(_ion)
            _unmapped = _all_grid_ions - set(ion_to_block.keys())
            if _unmapped:
                logger.warning(
                    "[BlockPurity] %d ions in grid NOT in ion_to_block: %s",
                    len(_unmapped), sorted(_unmapped),
                )
        else:
            logger.debug(
                "[BlockPurity] INITIAL LAYOUT PURE — all %d ions in home blocks",
                len(ion_to_block),
            )

    # ── Nested helpers (H2/H3/H4): shared EC and gadget routing ──

    def _route_ec_fresh(
        phase_pairs: list,
        n_pairs: int,
        start_layout: np.ndarray,
    ) -> Tuple[list, np.ndarray]:
        """Route an EC phase fresh with per-stabilizer-round splitting.

        Uses per-block decomposition when multi-block, otherwise routes
        on the full grid.  Captures outer scope variables.

        Returns ``(phase_steps, final_layout)``.
        """
        stab_round_size = ms_rounds_per_ec_round or cx_per_ec_round or n_pairs
        num_stab_rounds = max(1, n_pairs // stab_round_size)
        result_steps: list = []
        current = np.array(start_layout, copy=True)
        intra_offset = 0

        for sr_idx in range(num_stab_rounds):
            sr_start = sr_idx * stab_round_size
            sr_end = min(sr_start + stab_round_size, n_pairs)
            sr_pairs = phase_pairs[sr_start:sr_end]
            if not sr_pairs:
                continue
            sr_pair_count = len(sr_pairs)

            if _use_per_block:
                # --- Per-block decomposition (P5 fix: cross-block
                # pairs route on merged grid, not single block) ---
                block_pairs_map: Dict[str, List[List[Tuple[int, int]]]] = {
                    bname: [[] for _ in range(sr_pair_count)]
                    for bname in block_sub_grids
                }
                # Collect cross-block pairs separately — these need
                # the merged bounding-box grid, not a single block's
                # sub-grid.
                cross_block_pairs: List[List[Tuple[int, int]]] = [
                    [] for _ in range(sr_pair_count)
                ]
                _has_cross_block = False
                _cross_block_blocks: set = set()

                for round_i, pairs_in_round in enumerate(sr_pairs):
                    for pair in pairs_in_round:
                        a_idx, d_idx = pair
                        blk_a = ion_to_block.get(a_idx)
                        blk_d = ion_to_block.get(d_idx)
                        if blk_a == blk_d and blk_a is not None:
                            # Both ions in same block — route per-block
                            block_pairs_map[blk_a][round_i].append(pair)
                        elif blk_a is not None and blk_d is not None and blk_a != blk_d:
                            # Cross-block pair — must route on merged grid
                            cross_block_pairs[round_i].append(pair)
                            _has_cross_block = True
                            _cross_block_blocks.add(blk_a)
                            _cross_block_blocks.add(blk_d)
                        elif blk_a is not None and blk_d is None:
                            # Ion d unknown block (bridge ancilla?) —
                            # route on merged grid for safety
                            cross_block_pairs[round_i].append(pair)
                            _has_cross_block = True
                            _cross_block_blocks.add(blk_a)
                        elif blk_d is not None and blk_a is None:
                            cross_block_pairs[round_i].append(pair)
                            _has_cross_block = True
                            _cross_block_blocks.add(blk_d)
                        else:
                            # Both ions have unknown block — fallback
                            default_block = list(block_sub_grids.keys())[0]
                            block_pairs_map[default_block][round_i].append(pair)

                if _has_cross_block:
                    _n_xb = sum(len(r) for r in cross_block_pairs)
                    logger.info(
                        "[PhaseSteps] EC phase: %d cross-block pairs "
                        "detected across blocks %s — routing on merged grid",
                        _n_xb, sorted(_cross_block_blocks),
                    )

                per_block_steps: Dict[str, list] = {}
                for bname, sg in block_sub_grids.items():
                    r0, c0, r1, c1 = sg.grid_region
                    c0i, c1i = c0 * k, c1 * k
                    block_rows = r1 - r0
                    block_cols = c1 - c0

                    bp = block_pairs_map.get(bname, [])
                    if not any(bp):
                        per_block_steps[bname] = []
                        continue

                    block_layout = current[r0:r1, c0i:c1i].copy()
                    block_active = [
                        idx for idx in sg.ion_indices
                        if idx in active_ions_set
                    ]
                    block_arch = QCCDWiseArch(
                        m=block_cols, n=block_rows, k=k,
                    )
                    # --- Fix A: Route MS rounds WITHOUT bundled return ---
                    # The return round is now solved as a separate SAT
                    # instance after the MS rounds, reducing R by 1 in
                    # every SAT call and avoiding UNSAT from conflicting
                    # return-BT + gate-pairing constraints.
                    block_p_arr = list(bp)
                    # Blocks are routed sequentially in this loop,
                    # so there is no nested parallelism concern —
                    # pass through the real max_inner_workers to allow
                    # the SAT pool to use multiple workers.
                    blk_steps, _blk_final = _route_round_sequence(
                        block_layout,
                        block_arch,
                        block_p_arr,
                        lookahead=min(lookahead, len(block_p_arr)),
                        subgridsize=subgridsize,
                        base_pmax_in=base_pmax_in or 1,
                        active_ions=block_active,
                        stop_event=stop_event,
                        max_inner_workers=max_inner_workers,
                        progress_callback=_phase_cb,
                        replay_level=replay_level,
                        ms_rounds_per_ec_round=ms_rounds_per_ec_round or 4,
                        max_sat_time=max_sat_time,
                        max_rc2_time=max_rc2_time,
                        solver_params=solver_params,
                    )
                    # When replay_level == 0 we skip per-block return
                    # reconfig entirely — no route-back at all.
                    # When replay_level > 0 we defer route-back to a
                    # single batched full-grid call after merging all
                    # blocks (see below).
                    per_block_steps[bname] = list(blk_steps)

                sr_merged, current = _merge_block_routing_steps(
                    per_block_steps, block_sub_grids,
                    current, k,
                    wiseArch=wiseArch,
                    subgridsize=subgridsize,
                    base_pmax_in=base_pmax_in,
                    stop_event=stop_event,
                    max_inner_workers=max_inner_workers,
                    max_sat_time=max_sat_time,
                    max_rc2_time=max_rc2_time,
                    allow_heuristic_fallback=heuristic_fallback_for_noncache,
                    solver_params=solver_params,
                )

                # ── Batched full-grid return reconfig ──
                # Instead of returning ions per-block independently,
                # compute a single full-grid return reconfig across
                # all blocks.  This coordinates ion movements at block
                # boundaries and reduces total swap cost.
                # When replay_level >= num_stab_rounds (the "d" case),
                # only route-back at the end of the full EC block, not
                # after each stabilizer round.
                # When cache_ec_rounds is False, skip return entirely —
                # there is no cached schedule to replay, so routing
                # ions back to a starting layout is pure overhead.
                _do_per_round_return = (
                    cache_ec_rounds
                    and replay_level > 0
                    and replay_level < num_stab_rounds
                    and not np.array_equal(current, start_layout)
                )
                if _do_per_round_return:
                    _return_steps = _compute_return_reconfig(
                        current, start_layout,
                        wiseArch, subgridsize,
                        base_pmax_in=base_pmax_in or 1,
                        stop_event=stop_event,
                        max_inner_workers=max_inner_workers,
                        progress_callback=_phase_cb,
                        allow_heuristic_fallback=True,
                        # Always skip SAT for return reconfigs — these
                        # are pure layout permutations where the heuristic
                        # (odd-even sort) always succeeds instantly.
                        use_heuristic_directly=True,
                        max_sat_time=max_sat_time,
                        max_rc2_time=max_rc2_time,
                        solver_params=solver_params,
                    )
                    sr_merged.extend(_return_steps)
                    if _return_steps:
                        current = _return_steps[-1].layout_after.copy()

                # --- P5: Route cross-block pairs on merged bbox grid ---
                if _has_cross_block:
                    _xb_nonempty = [r for r in cross_block_pairs if r]
                    if _xb_nonempty:
                        # Build bounding box covering all involved blocks
                        _xb_sgs = [
                            block_sub_grids[nm]
                            for nm in sorted(_cross_block_blocks)
                            if nm in block_sub_grids
                        ]
                        _PAD_R = 1
                        _PAD_C = 1
                        _xb_r0 = max(0, min(
                            sg.grid_region[0] for sg in _xb_sgs
                        ) - _PAD_R)
                        _xb_r1 = min(n, max(
                            sg.grid_region[2] for sg in _xb_sgs
                        ) + _PAD_R)
                        _xb_c0 = max(0, min(
                            sg.grid_region[1] for sg in _xb_sgs
                        ) - _PAD_C)
                        _xb_c1 = min(m, max(
                            sg.grid_region[3] for sg in _xb_sgs
                        ) + _PAD_C)
                        _xb_c0p = _xb_c0 * k
                        _xb_c1p = _xb_c1 * k

                        _xb_layout = current[
                            _xb_r0:_xb_r1, _xb_c0p:_xb_c1p
                        ].copy()
                        _xb_rows = _xb_r1 - _xb_r0
                        _xb_m = _xb_c1 - _xb_c0
                        _xb_wise = QCCDWiseArch(
                            n=_xb_rows, m=_xb_m, k=k,
                        )
                        # Include ALL ions from ALL involved blocks as
                        # active so the SAT solver can shuffle them
                        # out of the way when routing cross-block
                        # pairs.  The merged bbox grid is tiled into
                        # subgridsize patches by _patch_and_route
                        # (called via _route_round_sequence), so
                        # individual SAT instances stay small.
                        _xb_active = []
                        for _sg in _xb_sgs:
                            _xb_active.extend(
                                idx for idx in _sg.ion_indices
                                if idx in active_ions_set
                            )
                        logger.debug(
                            "[PhaseSteps] cross-block SAT: bbox=%dx%d "
                            "(rows %d-%d, cols %d-%d), "
                            "active_ions=%d (of %d in blocks), "
                            "pairs=%d",
                            _xb_rows, _xb_m * k,
                            _xb_r0, _xb_r1, _xb_c0, _xb_c1,
                            len(_xb_active),
                            sum(len(sg.ion_indices) for sg in _xb_sgs),
                            sum(len(r) for r in _xb_nonempty),
                        )
                        _xb_p_arr = list(_xb_nonempty)

                        try:
                            _xb_steps, _xb_final = _route_round_sequence(
                                _xb_layout,
                                _xb_wise,
                                _xb_p_arr,
                                lookahead=min(
                                    lookahead, len(_xb_p_arr),
                                ),
                                subgridsize=subgridsize,
                                base_pmax_in=base_pmax_in or 1,
                                active_ions=_xb_active,
                                stop_event=stop_event,
                                max_inner_workers=max_inner_workers,
                                progress_callback=_phase_cb,
                                replay_level=replay_level,
                                ms_rounds_per_ec_round=ms_rounds_per_ec_round or 4,
                                max_sat_time=max_sat_time,
                                max_rc2_time=max_rc2_time,
                                solver_params=solver_params,
                            )
                            # P4: Cross-block route-back is deferred
                            # to a full-grid return below (after re-
                            # embedding), so that all blocks route back
                            # simultaneously on the full grid.
                            _xb_steps = list(_xb_steps)
                            # Re-embed bbox results into global layout
                            _sched_row_off = _xb_r0
                            _sched_col_off = _xb_c0p
                            for step in _xb_steps:
                                # Re-embed layout_before into global grid
                                if step.layout_before is not None:
                                    _gl_before = np.array(current, copy=True)
                                    _gl_before[
                                        _xb_r0:_xb_r1,
                                        _xb_c0p:_xb_c1p,
                                    ] = step.layout_before
                                    step.layout_before = _gl_before
                                else:
                                    step.layout_before = np.array(current, copy=True)
                                _gl = np.array(current, copy=True)
                                _gl[
                                    _xb_r0:_xb_r1,
                                    _xb_c0p:_xb_c1p,
                                ] = step.layout_after
                                step.layout_after = _gl
                                if step.schedule is not None:
                                    step.schedule = (
                                        _remap_schedule_to_global(
                                            step.schedule,
                                            _sched_row_off,
                                            _sched_col_off,
                                        )
                                    )
                                sr_merged.append(step)
                                current = _gl
                            # P4: Full-grid return after cross-block
                            # routing — all blocks route back together
                            # on the full grid (P5).
                            # Skip when cache_ec_rounds is False — no
                            # cached schedule needs a known start layout.
                            if (
                                cache_ec_rounds
                                and replay_level > 0
                                and not np.array_equal(
                                    current, start_layout,
                                )
                            ):
                                _xb_fg_return = (
                                    _compute_return_reconfig(
                                        current, start_layout,
                                        wiseArch, subgridsize,
                                        base_pmax_in=(
                                            base_pmax_in or 1
                                        ),
                                        stop_event=stop_event,
                                        max_inner_workers=(
                                            max_inner_workers
                                        ),
                                        progress_callback=_phase_cb,
                                        allow_heuristic_fallback=True,
                                        # Always skip SAT for return
                                        # reconfigs — pure permutations.
                                        use_heuristic_directly=True,
                                        max_sat_time=max_sat_time,
                                        max_rc2_time=max_rc2_time,
                                        solver_params=solver_params,
                                    )
                                )
                                sr_merged.extend(_xb_fg_return)
                                if _xb_fg_return:
                                    current = (
                                        _xb_fg_return[-1]
                                        .layout_after.copy()
                                    )
                            logger.info(
                                "[PhaseSteps] EC cross-block pairs "
                                "routed: %d steps on %dx%d merged grid",
                                len(list(_xb_steps)),
                                _xb_rows, _xb_m * k,
                            )
                        except Exception as _xb_exc:
                            logger.warning(
                                "[PhaseSteps] EC cross-block routing "
                                "failed (%s); pairs may be unrouted",
                                _xb_exc,
                            )
            else:
                # --- Single-grid fallback ---
                # Route MS rounds without bundled return round.
                sr_merged, _sr_final = _route_round_sequence(
                    np.array(current, copy=True),
                    wiseArch,
                    list(sr_pairs),
                    lookahead=min(lookahead, len(sr_pairs)),
                    subgridsize=subgridsize,
                    base_pmax_in=base_pmax_in or 1,
                    active_ions=active_ions,
                    stop_event=stop_event,
                    max_inner_workers=max_inner_workers,
                    progress_callback=_phase_cb,
                    replay_level=replay_level,
                    ms_rounds_per_ec_round=ms_rounds_per_ec_round or 4,
                    max_sat_time=max_sat_time,
                    max_rc2_time=max_rc2_time,
                    solver_params=solver_params,
                )
                # Separate return-round reconfig (skip when replay_level == 0
                # or when replay_level >= num_stab_rounds, i.e. route-back
                # only at EC block boundary).
                # Also skip when cache_ec_rounds is False — return serves
                # no purpose without a cache to replay into.
                _do_per_round_return = (
                    cache_ec_rounds
                    and replay_level > 0
                    and replay_level < num_stab_rounds
                )
                if _do_per_round_return:
                    _sr_return = _compute_return_reconfig(
                        _sr_final, current,
                        wiseArch, subgridsize,
                        base_pmax_in=base_pmax_in or 1,
                        stop_event=stop_event,
                        max_inner_workers=max_inner_workers,
                        progress_callback=_phase_cb,
                        allow_heuristic_fallback=True,
                        # Always skip SAT for return reconfigs.
                        use_heuristic_directly=True,
                        max_sat_time=max_sat_time,
                        max_rc2_time=max_rc2_time,
                        solver_params=solver_params,
                    )
                    sr_merged = list(sr_merged) + _sr_return
                else:
                    sr_merged = list(sr_merged)

            for step in sr_merged:
                if step.ms_round_index >= sr_pair_count:
                    step.ms_round_index = sr_pair_count - 1
                step.ms_round_index += intra_offset
            result_steps.extend(sr_merged)
            intra_offset += sr_pair_count

        # ── End-of-EC-block route-back REMOVED ──
        # The next phase's transition reconfig (or ec_cache replay
        # transition) already handles any layout difference, so an
        # explicit route-back here just wastes SAT calls.

        return result_steps, current

    def _apply_post_gadget_transition(
        phase_steps: list,
        cur_layout: np.ndarray,
        n_pairs: int,
        label: str,
    ) -> Tuple[list, np.ndarray]:
        """Add post-gadget transition reconfig if layout diverged from EC.

        Returns ``(updated_phase_steps, updated_layout)``.
        """
        if not ec_initial_layouts or not _use_per_block or not phase_steps:
            return phase_steps, cur_layout
        ec_target = _reconstruct_ec_target(
            ec_initial_layouts, cur_layout,
            block_sub_grids, k,
            ion_to_block=ion_to_block,
        )
        if np.array_equal(cur_layout, ec_target):
            return phase_steps, cur_layout
        _emit_reconfig_progress(
            f"Phase {phase_idx + 1}/{len(plans)}: returning ions to home blocks ({label})"
        )
        # Transition reconfig needs enough sorting passes (pmax) to
        # move ions across the grid.  Normal MS routing works with
        # pmax=1 because MS pairs guide the solver, but BT-only
        # reconfig must explicitly permute ions to distant targets.
        # Scale pmax with grid dimensions so ions can traverse the
        # full grid span in a single SAT cycle.
        _transition_pmax = max(base_pmax_in or 1, n, m * k, 3)
        _transition_skipped = False
        # Primary: per-block or full-grid SAT transition reconfig.
        # If this fails, try the simpler return-reconfig SAT approach.
        # If BOTH fail, gracefully continue with the current layout
        # — the EC routing in _route_ec_fresh already handles
        # cross-block pairs via the P5 merged-grid path, and
        # per-block pairs are classified by ion_to_block at route
        # time, so misplaced ions get routed on the merged grid
        # instead of a wrong block's sub-grid.
        transition: list = []
        try:
            try:
                transition = _compute_transition_reconfig_steps(
                    current_layout=cur_layout,
                    target_layout=ec_target,
                    wiseArch=wiseArch,
                    block_sub_grids=block_sub_grids,
                    ion_to_block=ion_to_block,
                    k=k,
                    subgridsize=subgridsize,
                    base_pmax_in=_transition_pmax,
                    max_inner_workers=max_inner_workers,
                    stop_event=stop_event,
                    progress_callback=_phase_cb,
                    # ── HEURISTIC FALLBACK POLICY ────────────────
                    # Transition reconfigs are pure layout permutations
                    # (NOT MS-gate routing).  The odd-even transposition
                    # sort heuristic is mathematically guaranteed to
                    # produce the exact target for any valid permutation.
                    # Always allow heuristic fallback to prevent
                    # unbounded SAT escalation stalls.
                    # ───────────────────────────────────────────────
                    allow_heuristic_fallback=True,
                    # Always skip SAT for transition reconfigs — these
                    # are pure layout permutations where the heuristic
                    # (odd-even sort) always succeeds instantly.
                    use_heuristic_directly=True,
                    max_sat_time=max_sat_time,
                    max_rc2_time=max_rc2_time,
                    solver_params=solver_params,
                )
            except Exception as exc:
                logger.warning(
                    "[PhaseSteps] gadget phase %d (%s): transition "
                    "reconfig failed (%s); retrying with direct "
                    "return-round SAT",
                    phase_idx, label, exc,
                )
                transition = _compute_return_reconfig(
                    cur_layout, ec_target,
                    wiseArch, subgridsize,
                    base_pmax_in=_transition_pmax,
                    stop_event=stop_event,
                    max_inner_workers=max_inner_workers,
                    progress_callback=_phase_cb,
                    allow_heuristic_fallback=True,
                    use_heuristic_directly=True,
                    max_sat_time=max_sat_time,
                    max_rc2_time=max_rc2_time,
                    solver_params=solver_params,
                )
        except Exception as exc_final:
            # FM5 fix: only swallow the exception if all ions happen to
            # be in their home blocks already (transition was cosmetic).
            # If ions are genuinely misplaced, raise immediately with
            # full diagnostics instead of silently continuing.
            _misplaced_count = 0
            if ion_to_block and block_sub_grids:
                _fm5_regions: Dict[str, Tuple[int, int, int, int]] = {}
                for _bname, _sg in block_sub_grids.items():
                    _r0, _c0, _r1, _c1 = _sg.grid_region
                    _fm5_regions[_bname] = (_r0, _c0 * k, _r1, _c1 * k)
                for _r in range(cur_layout.shape[0]):
                    for _c in range(cur_layout.shape[1]):
                        _ion = int(cur_layout[_r, _c])
                        if _ion == 0 or _ion not in ion_to_block:
                            continue
                        _home = ion_to_block[_ion]
                        _hr0, _hc0, _hr1, _hc1 = _fm5_regions.get(
                            _home, (0, 0, 0, 0))
                        if not (_hr0 <= _r < _hr1 and _hc0 <= _c < _hc1):
                            _misplaced_count += 1

            if _misplaced_count > 0:
                raise ValueError(
                    f"[PhaseSteps] gadget phase {phase_idx} ({label}): "
                    f"BOTH transition reconfig attempts failed "
                    f"({exc_final}). {_misplaced_count} ions are in "
                    f"non-home blocks. Cannot continue — EC routing "
                    f"would produce incorrect circuits."
                ) from exc_final
            else:
                logger.warning(
                    "[PhaseSteps] gadget phase %d (%s): transition "
                    "reconfig failed (%s) but all ions are in home "
                    "blocks — continuing.",
                    phase_idx, label, exc_final,
                )
                transition = []
                _transition_skipped = True
        for ts in transition:
            ts.ms_round_index = n_pairs
        phase_steps.extend(transition)
        if transition:
            cur_layout = transition[-1].layout_after.copy()
        logger.info(
            "[PhaseSteps] gadget phase %d (%s): post-gadget "
            "transition reconfig: %d steps",
            phase_idx, label, len(transition),
        )

        # ── P6: Post-gadget route-back validation ──
        # Verify every ion is back in its home block region.
        # Log any stragglers so we can detect incomplete route-back.
        if ion_to_block and block_sub_grids:
            _misplaced: list = []
            # Build block → grid region lookup
            _block_regions: Dict[str, Tuple[int, int, int, int]] = {}
            for _bname, _sg in block_sub_grids.items():
                _r0, _c0, _r1, _c1 = _sg.grid_region
                _block_regions[_bname] = (_r0, _c0 * k, _r1, _c1 * k)

            for _r in range(cur_layout.shape[0]):
                for _c in range(cur_layout.shape[1]):
                    _ion = int(cur_layout[_r, _c])
                    if _ion == 0:
                        continue
                    _home = ion_to_block.get(_ion)
                    if _home is None:
                        continue  # spectator or unmapped ion
                    _hr0, _hc0, _hr1, _hc1 = _block_regions.get(
                        _home, (0, 0, 0, 0),
                    )
                    if not (_hr0 <= _r < _hr1 and _hc0 <= _c < _hc1):
                        _misplaced.append((_ion, _home, _r, _c))

            if _misplaced:
                _msg = (
                    f"[PhaseSteps] gadget phase {phase_idx} ({label}): "
                    f"P6 route-back validation: {len(_misplaced)} ions "
                    f"NOT in home block after transition! "
                    f"Misplaced: {_misplaced[:10]}"
                )
                if _transition_skipped:
                    # Transition already failed — downgrade to warning
                    # and let EC routing handle via merged-grid path (P5)
                    logger.warning(_msg)
                else:
                    logger.error(_msg)
                    raise RuntimeError(_msg)
            else:
                logger.debug(
                    "[PhaseSteps] gadget phase %d (%s): P6 route-back "
                    "validation: all ions in home blocks ✓",
                    phase_idx, label,
                )

        return phase_steps, cur_layout

    for phase_idx, plan in enumerate(plans):
        n_pairs = len(plan.ms_pairs_per_round)
        if n_pairs <= 0:
            continue

        phase_pairs = plan.ms_pairs_per_round

        # Merge adjacent rounds with disjoint ion sets — this lets
        # cross-block EC CX execute in the same routing round, reducing
        # the total number of reconfiguration cycles.
        phase_pairs = _merge_phase_pairs(phase_pairs)
        n_pairs = len(phase_pairs)

        # ── Block purity gate: assert all ions in home blocks ────
        # After a return-reconfig (phase_idx > 0) every ion MUST be
        # back inside its owning block's sub-grid.  A violation here
        # means the previous return-reconfig failed to converge and
        # per-block routing would silently produce corrupted layouts
        # (duplicate / missing ions after merge overlay).
        if phase_idx > 0 and ion_to_block and _use_per_block:
            _entry_misplaced: Dict[str, List[int]] = {}
            for _bname, _sg in block_sub_grids.items():
                _r0, _c0, _r1, _c1 = _sg.grid_region
                _c0i, _c1i = _c0 * k, _c1 * k
                _blk_slice = current_layout[_r0:_r1, _c0i:_c1i]
                for _ri in range(_blk_slice.shape[0]):
                    for _ci in range(_blk_slice.shape[1]):
                        _ion = int(_blk_slice[_ri, _ci])
                        if (
                            _ion != 0
                            and _ion in ion_to_block
                            and ion_to_block[_ion] != _bname
                        ):
                            _entry_misplaced.setdefault(_bname, []).append(_ion)
            if _entry_misplaced:
                _pt_label = getattr(plan.phase_type, 'name', plan.phase_type)
                raise ValueError(
                    f"[BlockPurity] Phase {phase_idx + 1}/{len(plans)} "
                    f"({_pt_label}): ions in wrong blocks at phase entry "
                    f"(return-reconfig failed to converge). "
                    f"Misplaced: {({b: sorted(ions) for b, ions in _entry_misplaced.items()})}. "
                    f"This means the previous phase's route-back did not "
                    f"restore all ions to their home sub-grids."
                )
            else:
                logger.debug(
                    "[BlockPurity] Phase %d/%d: entry check PASSED — "
                    "all %d ions in home blocks",
                    phase_idx + 1, len(plans), len(ion_to_block),
                )

        # =============================================================
        # EC Phase
        # =============================================================
        _is_ec_phase = _is_ec_phase_type(plan.phase_type)
        if _is_ec_phase:
            # --- Cache check ---
            # Use total pair count (not n_pairs = merged round count)
            # in cache key so phases with different numbers of actual
            # MS pairs don't share cached steps.  Phase 0 (pre-EC, 48
            # pairs) and Phase 6 (post-EC, 32 pairs) may share the
            # same round_signature but must NOT share cache entries.
            _phase_pair_total = sum(
                len(r) for r in plan.ms_pairs_per_round
            ) if plan.ms_pairs_per_round else 0
            _ec_cache_key = (
                (plan.round_signature, _phase_pair_total)
                if plan.round_signature is not None
                else None
            )
            if (
                cache_ec_rounds
                and _ec_cache_key is not None
                and _ec_cache_key in ec_cache
            ):
                cached_starting_layout, cached_steps = ec_cache[_ec_cache_key]
                phase_steps: List[RoutingStep] = []
                _cache_replayed = True

                # If the current layout differs from the cached starting
                # layout, prepend a transition reconfig to return us to
                # the layout the cached schedule expects.
                if not np.array_equal(current_layout, cached_starting_layout):
                    _emit_phase(
                        phase_idx,
                        "Routing to cached EC layout",
                    )
                    _emit_reconfig_progress(
                        f"Phase {phase_idx + 1}/{len(plans)} (EC cached): "
                        f"transition reconfig to cached layout"
                    )
                    try:
                        transition_steps = _compute_transition_reconfig_steps(
                            current_layout=current_layout,
                            target_layout=cached_starting_layout,
                            wiseArch=wiseArch,
                            block_sub_grids=block_sub_grids,
                            ion_to_block=ion_to_block,
                            k=k,
                            subgridsize=subgridsize,
                            base_pmax_in=base_pmax_in or 1,
                            max_inner_workers=max_inner_workers,
                            stop_event=stop_event,
                            progress_callback=_phase_cb,
                            allow_heuristic_fallback=True,
                            use_heuristic_directly=True,
                            max_sat_time=max_sat_time,
                            max_rc2_time=max_rc2_time,
                            solver_params=solver_params,
                        )
                    except Exception as exc:
                        logger.warning(
                            "[PhaseSteps] phase=%d (EC cached): transition "
                            "reconfig failed (%s); routing fresh instead",
                            phase_idx, exc,
                        )
                        transition_steps = []
                    for ts in transition_steps:
                        ts.ms_round_index = 0
                        ts.from_cache = False
                        ts.can_merge_with_next = True
                        ts.is_initial_placement = False
                    phase_steps.extend(transition_steps)
                    if transition_steps:
                        current_layout = transition_steps[-1].layout_after.copy()
                        logger.info(
                            "[PhaseSteps] phase=%d (EC cached): SAT-based "
                            "transition reconfig: %d steps",
                            phase_idx, len(transition_steps),
                        )

                    # C1 fix: If transition didn't converge, route fresh
                    # instead of replaying stale schedules.  Previous code
                    # used ``continue`` which skipped global bookkeeping
                    # (ms_round_index renumbering, all_routing_steps.extend,
                    # round_cursor advancement) — silently losing steps.
                    if not np.array_equal(current_layout, cached_starting_layout):
                        _cache_replayed = False
                        logger.warning(
                            "[PhaseSteps] phase=%d (EC cached): transition "
                            "did NOT converge (%d cells remain); re-routing fresh.",
                            phase_idx,
                            int(np.count_nonzero(current_layout != cached_starting_layout)),
                        )
                        fresh_steps, current_layout = _route_ec_fresh(
                            phase_pairs, n_pairs, current_layout,
                        )
                        phase_steps.extend(fresh_steps)

                if _cache_replayed:
                    _emit_phase(
                        phase_idx,
                        "Replaying cached EC phase",
                    )
                    # Replay cached steps.
                    #
                    # Change 2 (P2, P6): For the first step, attempt a
                    # SAT-based schedule rebuild from current_layout →
                    # step.layout_after.  This avoids the unconditional
                    # heuristic fallback of the old approach (schedule=None).
                    # If SAT can't converge, fall back to schedule=None
                    # (heuristic is acceptable for cache_replay per P2).
                    #
                    # Subsequent steps chain correctly because step N's
                    # layout_after == step N+1's expected starting layout.
                    _replay_layout = np.array(current_layout, copy=True)
                    for _ci, step in enumerate(cached_steps):
                        if _ci == 0:
                            # ── SAT-first for first cached step ──
                            _first_target = np.array(
                                step.layout_after, copy=True,
                            )
                            if np.array_equal(_replay_layout, _first_target):
                                # Layouts match — use cached schedule
                                _sched = (
                                    [dict(p) for p in step.schedule]
                                    if step.schedule is not None
                                    else None
                                )
                            else:
                                # Try SAT from current → first target
                                # (skip SAT if heuristic_cache_replay)
                                _sat_snaps = []
                                if heuristic_cache_replay:
                                    _sched = None
                                    logger.info(
                                        "[PhaseSteps] phase=%d (EC cached): "
                                        "heuristic_cache_replay=True — using "
                                        "heuristic for first step",
                                        phase_idx,
                                    )
                                else:
                                    try:
                                        _sat_snaps = _rebuild_schedule_for_layout(
                                            np.array(_replay_layout, copy=True),
                                            wiseArch,
                                            _first_target,
                                            subgridsize=subgridsize,
                                            base_pmax_in=base_pmax_in or 1,
                                            stop_event=stop_event,
                                            max_inner_workers=max_inner_workers,
                                            allow_heuristic_fallback=True,
                                            max_sat_time=max_sat_time,
                                            max_rc2_time=max_rc2_time,
                                            solver_params=solver_params,
                                        )
                                    except Exception as _sat_exc:
                                        logger.warning(
                                            "[PhaseSteps] phase=%d (EC cached): "
                                            "SAT rebuild for first step failed: %s",
                                            phase_idx, _sat_exc,
                                        )
                                        _sat_snaps = []
                                if _sat_snaps:
                                    # Insert intermediate SAT cycles as
                                    # extra cache_replay steps.
                                    for _sl, _ss, _ in _sat_snaps[:-1]:
                                        phase_steps.append(RoutingStep(
                                            layout_after=np.array(_sl, copy=True),
                                            schedule=_ss,
                                            solved_pairs=[],
                                            ms_round_index=-1,
                                            from_cache=True,
                                            tiling_meta=(0, 0),
                                            can_merge_with_next=False,
                                            is_layout_transition=True,
                                            layout_before=np.array(
                                                _replay_layout, copy=True,
                                            ),
                                            reconfig_context="cache_replay",
                                        ))
                                        _replay_layout = np.array(
                                            _sl, copy=True,
                                        )
                                    _sched = _sat_snaps[-1][1]
                                    logger.info(
                                        "[PhaseSteps] phase=%d (EC cached): "
                                        "SAT rebuilt first step via %d cycle(s)",
                                        phase_idx, len(_sat_snaps),
                                    )
                                else:
                                    # Fall back to None → heuristic (P2)
                                    _sched = None
                                    logger.info(
                                        "[PhaseSteps] phase=%d (EC cached): "
                                        "SAT rebuild returned 0 snapshots; "
                                        "falling back to heuristic for first step",
                                        phase_idx,
                                    )
                        else:
                            # Subsequent steps: use cached schedule
                            _sched = (
                                [dict(p) for p in step.schedule]
                                if step.schedule is not None
                                else None
                            )
                        phase_steps.append(RoutingStep(
                            layout_after=np.array(step.layout_after, copy=True),
                            schedule=_sched,
                            solved_pairs=step.solved_pairs,
                            ms_round_index=step.ms_round_index,
                            from_cache=True,
                            tiling_meta=step.tiling_meta,
                            can_merge_with_next=step.can_merge_with_next,
                            is_initial_placement=False,
                            layout_before=np.array(_replay_layout, copy=True),
                            reconfig_context="cache_replay",
                        ))
                        _replay_layout = np.array(step.layout_after, copy=True)
                    logger.info(
                        "[PhaseSteps] phase=%d (EC cached): replaying %d steps",
                        phase_idx, len(phase_steps),
                    )
                # Keep current_layout in sync with replayed steps so
                # that downstream phases (gadget, EC post) see the
                # correct starting layout.
                if phase_steps:
                    current_layout = np.array(
                        phase_steps[-1].layout_after, copy=True,
                    )
            else:
                # --- Per-stabilizer-round splitting (H2: unified helper) ---
                _emit_phase(
                    phase_idx,
                    "SAT solving EC phase",
                )
                ec_starting_layout = np.array(current_layout, copy=True)
                phase_steps, current_layout = _route_ec_fresh(
                    phase_pairs, n_pairs, current_layout,
                )

                # Cache for future identical phases (with starting layout)
                if cache_ec_rounds and _ec_cache_key is not None:
                    ec_cache[_ec_cache_key] = (
                        ec_starting_layout, phase_steps,
                    )

            # Update per-block EC layouts for gadget exit BT
            if _use_per_block:
                for bname, sg in block_sub_grids.items():
                    r0, c0, r1, c1 = sg.grid_region
                    c0i, c1i = c0 * k, c1 * k
                    ec_initial_layouts[bname] = (
                        current_layout[r0:r1, c0i:c1i].copy()
                    )

        # =============================================================
        # Gadget Phase
        # =============================================================
        elif plan.phase_type == 'gadget':
            phase_pairs = plan.ms_pairs_per_round or []

            phase_pairs, n_pairs = _preprocess_gadget_pairs(
                phase_pairs, ion_to_block, phase_idx, logger,
            )

            # ── No-MS gadget phases (prep, measurement, single-block) ──
            # Skip SAT routing entirely; no routing step needed since
            # the layout doesn't change.
            if not phase_pairs or not any(phase_pairs):
                logger.debug(
                    "[PhaseSteps] gadget phase %d: no MS pairs "
                    "(prep/meas/single-block) — skip routing",
                    phase_idx,
                )
                # No RoutingStep emitted — layout unchanged.
                continue

            # Fix B Part 2: Check gadget-phase cache before routing
            _gadget_cache_hit = False
            _gadget_pair_total = sum(
                len(r) for r in phase_pairs
            ) if phase_pairs else 0
            _gadget_cache_key = (
                (plan.round_signature, _gadget_pair_total)
                if plan.round_signature is not None
                else None
            )
            if (
                cache_ec_rounds
                and _gadget_cache_key is not None
                and _gadget_cache_key in ec_cache
            ):
                cached_starting_layout, cached_steps = ec_cache[_gadget_cache_key]
                phase_steps: List[RoutingStep] = []

                if not np.array_equal(current_layout, cached_starting_layout):
                    _emit_reconfig_progress(
                        f"Phase {phase_idx + 1}/{len(plans)} (gadget cached): "
                        f"transition reconfig to cached layout"
                    )
                    try:
                        transition_steps = _compute_transition_reconfig_steps(
                            current_layout=current_layout,
                            target_layout=cached_starting_layout,
                            wiseArch=wiseArch,
                            block_sub_grids=block_sub_grids,
                            ion_to_block=ion_to_block,
                            k=k,
                            subgridsize=subgridsize,
                            base_pmax_in=base_pmax_in or 1,
                            max_inner_workers=max_inner_workers,
                            stop_event=stop_event,
                            progress_callback=_phase_cb,
                            allow_heuristic_fallback=True,
                            use_heuristic_directly=True,
                            max_sat_time=max_sat_time,
                            max_rc2_time=max_rc2_time,
                            solver_params=solver_params,
                        )
                    except Exception as exc:
                        logger.warning(
                            "[PhaseSteps] phase=%d (gadget cached): transition "
                            "reconfig failed (%s); routing fresh instead",
                            phase_idx, exc,
                        )
                        transition_steps = []
                    for ts in transition_steps:
                        ts.ms_round_index = 0
                        ts.from_cache = False
                        ts.can_merge_with_next = True
                        ts.is_initial_placement = False
                    phase_steps.extend(transition_steps)
                    if transition_steps:
                        current_layout = transition_steps[-1].layout_after.copy()

                    if not np.array_equal(current_layout, cached_starting_layout):
                        logger.warning(
                            "[PhaseSteps] phase=%d (gadget cached): "
                            "transition did NOT converge; re-routing fresh.",
                            phase_idx,
                        )
                    else:
                        _gadget_cache_hit = True
                else:
                    _gadget_cache_hit = True

                if _gadget_cache_hit:
                    # Change 2 (P2, P6): SAT-first replay for gadget
                    # cache — same logic as EC cache replay above.
                    _replay_layout = np.array(current_layout, copy=True)
                    for _ci, step in enumerate(cached_steps):
                        if _ci == 0:
                            # ── SAT-first for first cached step ──
                            _first_target = np.array(
                                step.layout_after, copy=True,
                            )
                            if np.array_equal(_replay_layout, _first_target):
                                _sched = (
                                    [dict(p) for p in step.schedule]
                                    if step.schedule is not None
                                    else None
                                )
                            else:
                                # Skip SAT if heuristic_cache_replay
                                if heuristic_cache_replay:
                                    _sched = None
                                    logger.info(
                                        "[PhaseSteps] phase=%d (gadget cached): "
                                        "heuristic_cache_replay=True — using "
                                        "heuristic for first step",
                                        phase_idx,
                                    )
                                else:
                                    try:
                                        _sat_snaps = _rebuild_schedule_for_layout(
                                            np.array(_replay_layout, copy=True),
                                            wiseArch,
                                            _first_target,
                                            subgridsize=subgridsize,
                                            base_pmax_in=base_pmax_in or 1,
                                            stop_event=stop_event,
                                            max_inner_workers=max_inner_workers,
                                            allow_heuristic_fallback=True,
                                            max_sat_time=max_sat_time,
                                            max_rc2_time=max_rc2_time,
                                            solver_params=solver_params,
                                        )
                                    except Exception as _sat_exc:
                                        logger.warning(
                                            "[PhaseSteps] phase=%d (gadget cached): "
                                            "SAT rebuild for first step failed: %s",
                                            phase_idx, _sat_exc,
                                        )
                                        _sat_snaps = []
                                if _sat_snaps:
                                    for _sl, _ss, _ in _sat_snaps[:-1]:
                                        phase_steps.append(RoutingStep(
                                            layout_after=np.array(
                                                _sl, copy=True,
                                            ),
                                            schedule=_ss,
                                            solved_pairs=[],
                                            ms_round_index=-1,
                                            from_cache=True,
                                            tiling_meta=(0, 0),
                                            can_merge_with_next=False,
                                            is_layout_transition=True,
                                            layout_before=np.array(
                                                _replay_layout, copy=True,
                                            ),
                                            reconfig_context="cache_replay",
                                        ))
                                        _replay_layout = np.array(
                                            _sl, copy=True,
                                        )
                                    _sched = _sat_snaps[-1][1]
                                    logger.info(
                                        "[PhaseSteps] phase=%d (gadget cached): "
                                        "SAT rebuilt first step via %d cycle(s)",
                                        phase_idx, len(_sat_snaps),
                                    )
                                else:
                                    _sched = None
                                    logger.info(
                                        "[PhaseSteps] phase=%d (gadget cached): "
                                        "SAT rebuild returned 0 snapshots; "
                                        "falling back to heuristic for first step",
                                        phase_idx,
                                    )
                        else:
                            _sched = (
                                [dict(p) for p in step.schedule]
                                if step.schedule is not None
                                else None
                            )
                        phase_steps.append(RoutingStep(
                            layout_after=np.array(step.layout_after, copy=True),
                            schedule=_sched,
                            solved_pairs=step.solved_pairs,
                            ms_round_index=step.ms_round_index,
                            from_cache=True,
                            tiling_meta=step.tiling_meta,
                            can_merge_with_next=step.can_merge_with_next,
                            is_initial_placement=False,
                            layout_before=np.array(_replay_layout, copy=True),
                            reconfig_context="cache_replay",
                        ))
                        _replay_layout = np.array(step.layout_after, copy=True)
                    if phase_steps:
                        current_layout = np.array(
                            phase_steps[-1].layout_after, copy=True,
                        )
                    logger.info(
                        "[PhaseSteps] phase=%d (gadget cached): "
                        "replaying %d steps",
                        phase_idx, len(phase_steps),
                    )

            if not _gadget_cache_hit:
                _gadget_starting_layout = np.array(current_layout, copy=True)
                all_block_names = list(block_sub_grids.keys())
                interacting_names = plan.interacting_blocks or all_block_names
                interacting_sgs = [
                    block_sub_grids[nm]
                    for nm in interacting_names
                    if nm in block_sub_grids
                ]

                # ── Level-1 narrowing: restrict merged grid to bridge
                # blocks only.  When a gadget phase spans >2 blocks
                # (e.g. CSS Surgery merge phases list all 3 blocks for
                # interleaved EC), only the blocks connected by cross-
                # block (bridge) pairs need to share a merged grid.
                # Non-bridge blocks' intra-block EC pairs are routed
                # separately on their per-block sub-grids and overlaid.
                _non_bridge_blocks: list = []
                _non_bridge_ec_pairs: Dict[str, List[List[Tuple[int, int]]]] = {}
                if (
                    plan.phase_type == 'gadget'
                    and len(interacting_names) > 2
                    and ion_to_block
                ):
                    _bridge_blocks: set = set()
                    for _rnd in phase_pairs:
                        for _a, _d in _rnd:
                            _ba = ion_to_block.get(_a)
                            _bd = ion_to_block.get(_d)
                            if (
                                _ba is not None
                                and _bd is not None
                                and _ba != _bd
                            ):
                                _bridge_blocks.add(_ba)
                                _bridge_blocks.add(_bd)

                    if (
                        _bridge_blocks
                        and len(_bridge_blocks) < len(interacting_names)
                    ):
                        _non_bridge_blocks = [
                            nm for nm in interacting_names
                            if nm not in _bridge_blocks
                        ]
                        _bridge_ion_set: set = set()
                        for _nm in _bridge_blocks:
                            _bridge_ion_set.update(
                                block_sub_grids[_nm].ion_indices,
                            )

                        # Separate pairs into bridge-region and
                        # non-bridge per-block EC.
                        _narrowed_pairs: list = []
                        _non_bridge_ec_pairs = {
                            nm: [] for nm in _non_bridge_blocks
                        }
                        for _rnd in phase_pairs:
                            _br_rnd: list = []
                            _nb_rnds: Dict[str, list] = {
                                nm: [] for nm in _non_bridge_blocks
                            }
                            for _a, _d in _rnd:
                                _ba = ion_to_block.get(_a)
                                _bd = ion_to_block.get(_d)
                                if (
                                    _ba == _bd
                                    and _ba in _nb_rnds
                                ):
                                    _nb_rnds[_ba].append((_a, _d))
                                else:
                                    _br_rnd.append((_a, _d))
                            _narrowed_pairs.append(_br_rnd)
                            for _nm in _non_bridge_blocks:
                                _non_bridge_ec_pairs[_nm].append(
                                    _nb_rnds[_nm],
                                )

                        _orig_avg = (
                            sum(len(r) for r in phase_pairs)
                            / max(len(phase_pairs), 1)
                        )
                        _new_avg = (
                            sum(len(r) for r in _narrowed_pairs)
                            / max(len(_narrowed_pairs), 1)
                        )
                        logger.info(
                            "[PhaseSteps] phase=%d: narrowing merged "
                            "grid from %s to bridge blocks %s "
                            "(non-bridge EC: %s, pairs/round %.1f→%.1f)",
                            phase_idx,
                            interacting_names,
                            sorted(_bridge_blocks),
                            _non_bridge_blocks,
                            _orig_avg,
                            _new_avg,
                        )

                        phase_pairs = _narrowed_pairs
                        interacting_names = sorted(_bridge_blocks)
                        interacting_sgs = [
                            block_sub_grids[nm]
                            for nm in interacting_names
                            if nm in block_sub_grids
                        ]

                _use_level1_slicing = (
                    _use_per_block
                    and interacting_names is not None
                    and len(interacting_names) >= 1
                )

                # C2 fix: Compute _use_bbox once with all conditions merged.
                # Gadget phases always prefer bounding-box extraction to
                # preserve true spatial topology (Fix 8).  Non-gadget L1
                # phases only use bbox when blocks span multiple row bands
                # (Fix 5/6).
                _use_bbox = False
                if _use_level1_slicing and interacting_sgs:
                    if plan.phase_type == 'gadget':
                        _use_bbox = True
                    else:
                        _row_ranges = set(
                            (sg.grid_region[0], sg.grid_region[2])
                            for sg in interacting_sgs
                        )
                        if len(_row_ranges) > 1:
                            _use_bbox = True
                    if _use_bbox:
                        logger.debug(
                            "[PhaseSteps] gadget phase %d: using bounding-box "
                            "extraction (gadget=%s)",
                            phase_idx, plan.phase_type == 'gadget',
                        )

                if _use_level1_slicing and interacting_sgs:
                    # --- Level 1 slicing: merged sub-grid ---
                    # Build merged layout from CURRENT global layout.
                    # Two strategies depending on block arrangement:
                    #   (a) same row band → horizontal concatenation
                    #   (b) multi-row band → bounding-box extraction
                    block_regions: Dict[str, Tuple[int, int, int, int]] = {}

                    if _use_bbox:
                        # ── Fix 6: bounding-box extraction ──
                        # Extract the minimal rectangle from the global
                        # layout that covers all interacting blocks.  This
                        # preserves true spatial topology (unlike horizontal
                        # concatenation) and keeps the SAT problem smaller
                        # than the full grid for 2-block interactions.
                        #
                        # Fix 7: Add padding to bbox to ensure empty cells
                        # exist for ion movement.  A fully-packed bbox causes
                        # SAT intractability (no room for reconfiguration).
                        _PAD_ROWS = 1
                        _PAD_COLS = 1  # in trap columns
                        _bb_r0 = max(0, min(sg.grid_region[0] for sg in interacting_sgs) - _PAD_ROWS)
                        _bb_r1 = min(n, max(sg.grid_region[2] for sg in interacting_sgs) + _PAD_ROWS)
                        _bb_c0 = max(0, min(sg.grid_region[1] for sg in interacting_sgs) - _PAD_COLS)
                        _bb_c1 = min(m, max(sg.grid_region[3] for sg in interacting_sgs) + _PAD_COLS)
                        _bb_c0_phys = _bb_c0 * k
                        _bb_c1_phys = _bb_c1 * k

                        m_n_rows = _bb_r1 - _bb_r0
                        m_total_cols = _bb_c1_phys - _bb_c0_phys

                        merged_layout = current_layout[
                            _bb_r0:_bb_r1, _bb_c0_phys:_bb_c1_phys
                        ].copy()

                        for sg in interacting_sgs:
                            r0, c0, r1, c1 = sg.grid_region
                            block_regions[sg.block_name] = (
                                r0 - _bb_r0,
                                c0 * k - _bb_c0_phys,
                                r1 - _bb_r0,
                                c1 * k - _bb_c0_phys,
                            )

                        m_m = (_bb_c1 - _bb_c0)
                        logger.debug(
                            "[PhaseSteps] gadget phase %d: bbox extraction "
                            "%dx%d (rows %d-%d, trap cols %d-%d)",
                            phase_idx, m_n_rows, m_total_cols,
                            _bb_r0, _bb_r1, _bb_c0, _bb_c1,
                        )
                    else:
                        # ── Horizontal concatenation (same row band) ──
                        m_n_rows = max(sg.n_rows for sg in interacting_sgs)
                        col_offset = 0
                        m_total_cols = sum(sg.n_cols * k for sg in interacting_sgs)
                        merged_layout = np.zeros(
                            (m_n_rows, m_total_cols), dtype=int,
                        )

                        for sg in interacting_sgs:
                            r0, c0, r1, c1 = sg.grid_region
                            c0i = c0 * k
                            layout_cols = sg.n_cols * k
                            rows_to_copy = min(sg.n_rows, m_n_rows)
                            merged_layout[
                                :rows_to_copy,
                                col_offset:col_offset + layout_cols,
                            ] = current_layout[
                                r0:r0 + rows_to_copy, c0i:c0i + layout_cols
                            ]
                            block_regions[sg.block_name] = (
                                0, col_offset,
                                m_n_rows, col_offset + layout_cols,
                            )
                            col_offset += layout_cols

                        m_m = m_total_cols // k

                    merged_wise = QCCDWiseArch(n=m_n_rows, m=m_m, k=k)
                    # Issue 2 fix: Pass subgridsize unchanged to enable Level 2
                    # patch decomposition. The _patch_and_route function handles
                    # subgrids larger than the grid by clamping internally.
                    merged_sgs = subgridsize
                    merged_active = []
                    for sg in interacting_sgs:
                        merged_active.extend(
                            idx for idx in sg.ion_indices
                            if idx in active_ions_set
                        )

                    # Build exit BT (return ions to EC positions)
                    _bt_offsets = [
                        (sg.block_name, block_regions[sg.block_name][0],
                         block_regions[sg.block_name][1])
                        for sg in interacting_sgs
                    ]
                    # --- Fix A: Route gadget MS rounds without bundled
                    # exit-BT return round.  The return to EC positions
                    # is handled by _apply_post_gadget_transition which
                    # runs after this call and uses a separate SAT call.
                    phase_steps_raw, _mf = _route_round_sequence(
                        np.array(merged_layout, copy=True),
                        merged_wise,
                        list(phase_pairs),
                        lookahead=min(lookahead, len(phase_pairs)),
                        subgridsize=merged_sgs,
                        base_pmax_in=base_pmax_in or 1,
                        active_ions=merged_active,
                        stop_event=stop_event,
                        max_inner_workers=max_inner_workers,
                        progress_callback=_phase_cb,
                        replay_level=replay_level,
                        ms_rounds_per_ec_round=ms_rounds_per_ec_round or 4,
                        max_sat_time=max_sat_time,
                        max_rc2_time=max_rc2_time,
                        solver_params=solver_params,
                    )
                    phase_steps = list(phase_steps_raw)

                    # Group return round with last MS round
                    # (but preserve layout-transition steps in their own group)
                    for step in phase_steps:
                        if step.ms_round_index >= n_pairs and not step.is_layout_transition:
                            step.ms_round_index = n_pairs - 1

                    # Re-embed merged-grid results into global layout
                    # AND remap schedule swap coordinates from merged-sub-grid
                    # space to global ion-column space.  Without this remap,
                    # _runOddEvenReconfig applies swaps at wrong positions
                    # when the merged grid doesn't start at global (0, 0).
                    if _use_bbox:
                        # Bbox extraction: the merged grid IS a subgrid of
                        # the global layout.  Copy the entire bbox region
                        # back (including any empty cells where ions may
                        # have been routed through during SAT solving).
                        _sched_row_off = _bb_r0
                        _sched_col_off = _bb_c0_phys
                        for step in phase_steps:
                            global_after = np.array(current_layout, copy=True)
                            global_after[
                                _bb_r0:_bb_r1, _bb_c0_phys:_bb_c1_phys
                            ] = step.layout_after
                            step.layout_after = global_after
                            # Re-embed layout_before into global grid
                            # (without this, execution-loop pre-flight
                            # check crashes on shape mismatch).
                            if step.layout_before is not None:
                                global_before = np.array(
                                    current_layout, copy=True,
                                )
                                global_before[
                                    _bb_r0:_bb_r1,
                                    _bb_c0_phys:_bb_c1_phys,
                                ] = step.layout_before
                                step.layout_before = global_before
                            if step.schedule is not None:
                                step.schedule = _remap_schedule_to_global(
                                    step.schedule,
                                    _sched_row_off,
                                    _sched_col_off,
                                )
                            # ── Fix: advance base so next step's global
                            # layout includes this step's changes. Without
                            # this, subsequent steps use a stale base and
                            # ChainVerify / execution diverge (PRE-FLIGHT
                            # FAIL cascade).
                            current_layout = step.layout_after
                    else:
                        # Horizontal concat: blocks are repositioned in the
                        # merged grid, so re-embed block-by-block.
                        # The first interacting block's global position gives
                        # the offset (merged grid starts at row 0, col 0).
                        _sched_row_off = interacting_sgs[0].grid_region[0]
                        _sched_col_off = interacting_sgs[0].grid_region[1] * k
                        for step in phase_steps:
                            global_after = np.array(current_layout, copy=True)
                            for sg in interacting_sgs:
                                region = block_regions[sg.block_name]
                                r0_m, c0_m, r1_m, c1_m = region
                                r0_g, c0_g, r1_g, c1_g = sg.grid_region
                                c0_gi = c0_g * k
                                rows = min(r1_m - r0_m, r1_g - r0_g)
                                cols = min(c1_m - c0_m, (c1_g - c0_g) * k)
                                global_after[
                                    r0_g:r0_g + rows, c0_gi:c0_gi + cols
                                ] = step.layout_after[
                                    r0_m:r0_m + rows, c0_m:c0_m + cols
                                ]
                            step.layout_after = global_after
                            # Re-embed layout_before into global grid
                            if step.layout_before is not None:
                                global_before = np.array(
                                    current_layout, copy=True,
                                )
                                for sg in interacting_sgs:
                                    region = block_regions[sg.block_name]
                                    r0_m, c0_m, r1_m, c1_m = region
                                    r0_g, c0_g, r1_g, c1_g = sg.grid_region
                                    c0_gi = c0_g * k
                                    rows = min(r1_m - r0_m, r1_g - r0_g)
                                    cols = min(
                                        c1_m - c0_m, (c1_g - c0_g) * k,
                                    )
                                    global_before[
                                        r0_g:r0_g + rows,
                                        c0_gi:c0_gi + cols,
                                    ] = step.layout_before[
                                        r0_m:r0_m + rows,
                                        c0_m:c0_m + cols,
                                    ]
                                step.layout_before = global_before
                            if step.schedule is not None:
                                step.schedule = _remap_schedule_to_global(
                                    step.schedule,
                                    _sched_row_off,
                                    _sched_col_off,
                                )
                            # ── Fix: advance base so next step's global
                            # layout includes this step's changes (same
                            # rationale as bbox path above).
                            current_layout = step.layout_after

                    if phase_steps:
                        current_layout = phase_steps[-1].layout_after

                    phase_steps, current_layout = _apply_post_gadget_transition(
                        phase_steps, current_layout, n_pairs, "L1",
                    )

                    # ── Route non-bridge blocks' EC pairs per-block ──
                    # After bridge routing completes, route each non-
                    # bridge block's separated EC pairs independently on
                    # its own sub-grid, then overlay the layout changes
                    # onto current_layout.  This keeps each SAT call on
                    # a small per-block grid instead of the full merged
                    # bounding box.
                    if _non_bridge_ec_pairs:
                        # ── Fix 22: Renumber non-bridge EC step indices ──
                        # Non-bridge EC steps are routed independently on
                        # per-block sub-grids.  Their ms_round_index values
                        # from _route_round_sequence start at 0, which
                        # OVERLAPS with the bridge steps' already-assigned
                        # ms_round_index values.  After Fix 20 sorts all
                        # steps by ms_round_index, these overlapping values
                        # cause non-bridge and bridge steps to interleave,
                        # creating chain breaks that ChainVerify must fix
                        # with expensive SAT rebuilds (40+ rebuilds).
                        # Fix: offset non-bridge ms_round_index to sit
                        # AFTER all bridge steps and transition steps.
                        _nb_ms_offset = max(
                            (s.ms_round_index for s in phase_steps),
                            default=-1,
                        ) + 1

                        for _nb_name in _non_bridge_blocks:
                            _nb_rounds = _non_bridge_ec_pairs.get(
                                _nb_name, [],
                            )
                            _nb_nonempty = [
                                rnd for rnd in _nb_rounds if rnd
                            ]
                            if not _nb_nonempty:
                                continue
                            _nb_sg = block_sub_grids[_nb_name]
                            r0, c0, r1, c1 = _nb_sg.grid_region
                            c0i, c1i = c0 * k, c1 * k
                            _nb_layout = current_layout[
                                r0:r1, c0i:c1i
                            ].copy()
                            _nb_wise = QCCDWiseArch(
                                n=_nb_sg.n_rows,
                                m=_nb_sg.n_cols,
                                k=k,
                            )
                            _nb_active = [
                                idx for idx in _nb_sg.ion_indices
                                if idx in active_ions_set
                            ]
                            try:
                                _nb_steps, _ = _route_round_sequence(
                                    _nb_layout,
                                    _nb_wise,
                                    _nb_nonempty,
                                    lookahead=min(
                                        lookahead or 1,
                                        len(_nb_nonempty),
                                    ),
                                    subgridsize=subgridsize,
                                    base_pmax_in=base_pmax_in or 1,
                                    active_ions=_nb_active,
                                    stop_event=stop_event,
                                    max_inner_workers=(
                                        max_inner_workers
                                    ),
                                    progress_callback=_phase_cb,
                                    replay_level=replay_level,
                                    ms_rounds_per_ec_round=ms_rounds_per_ec_round or 4,
                                    max_sat_time=max_sat_time,
                                    max_rc2_time=max_rc2_time,
                                    solver_params=solver_params,
                                )
                                _nb_r0 = r0
                                _nb_c0 = c0i
                                for step in _nb_steps:
                                    _gl = np.array(
                                        current_layout, copy=True,
                                    )
                                    _gl[
                                        r0:r1, c0i:c1i
                                    ] = step.layout_after
                                    step.layout_after = _gl
                                    # Re-embed layout_before
                                    if step.layout_before is not None:
                                        _gl_b = np.array(
                                            current_layout, copy=True,
                                        )
                                        _gl_b[
                                            r0:r1, c0i:c1i
                                        ] = step.layout_before
                                        step.layout_before = _gl_b
                                    if step.schedule is not None:
                                        step.schedule = (
                                            _remap_schedule_to_global(
                                                step.schedule,
                                                _nb_r0,
                                                _nb_c0,
                                            )
                                        )
                                    # Fix 22: Renumber ms_round_index
                                    # so non-bridge EC steps sort AFTER
                                    # all bridge steps (prevents chain
                                    # interleaving after Fix 20 sort).
                                    step.ms_round_index += _nb_ms_offset
                                    phase_steps.append(step)
                                    current_layout = _gl
                                logger.info(
                                    "[PhaseSteps] phase=%d: non-bridge "
                                    "EC for %s routed %d steps on "
                                    "%dx%d grid",
                                    phase_idx,
                                    _nb_name,
                                    len(list(_nb_steps)),
                                    _nb_sg.n_rows,
                                    _nb_sg.n_cols * k,
                                )
                            except Exception as _exc:
                                logger.warning(
                                    "[PhaseSteps] phase=%d: non-bridge "
                                    "EC for %s failed: %s",
                                    phase_idx,
                                    _nb_name,
                                    _exc,
                                )

                else:
                    # --- Full-grid routing (all blocks active) ---
                    # Fix A: Route MS rounds directly without exit-BT.
                    # The return to EC positions is handled by
                    # _apply_post_gadget_transition below.
                    phase_steps_raw, current_layout = _route_round_sequence(
                        np.array(current_layout, copy=True),
                        wiseArch,
                        list(phase_pairs),
                        lookahead=min(
                            lookahead or 2, len(phase_pairs),
                        ),
                        subgridsize=subgridsize,
                        base_pmax_in=base_pmax_in or 1,
                        active_ions=active_ions,
                        stop_event=stop_event,
                        max_inner_workers=max_inner_workers,
                        progress_callback=_phase_cb,
                        replay_level=replay_level,
                        ms_rounds_per_ec_round=ms_rounds_per_ec_round or 4,
                        max_sat_time=max_sat_time,
                        max_rc2_time=max_rc2_time,
                        solver_params=solver_params,
                    )
                    phase_steps = list(phase_steps_raw)

                    # Group return round with last MS round
                    # (but preserve layout-transition steps in their own group)
                    for step in phase_steps:
                        if step.ms_round_index >= n_pairs and not step.is_layout_transition:
                            step.ms_round_index = n_pairs - 1

                    # H3: Reuse shared post-gadget transition helper
                    phase_steps, current_layout = _apply_post_gadget_transition(
                        phase_steps, current_layout, n_pairs, "full-grid",
                    )

                # Fix B: Cache fresh gadget routing for future phases
                if cache_ec_rounds and _gadget_cache_key is not None:
                    ec_cache[_gadget_cache_key] = (
                        _gadget_starting_layout, list(phase_steps),
                    )

        # =============================================================
        # Unknown phase type
        # =============================================================
        else:
            phase_steps_raw, current_layout = _route_round_sequence(
                np.array(current_layout, copy=True),
                wiseArch,
                phase_pairs,
                lookahead=min(lookahead, len(phase_pairs)),
                subgridsize=subgridsize,
                base_pmax_in=base_pmax_in or 1,
                active_ions=active_ions,
                stop_event=stop_event,
                max_inner_workers=max_inner_workers,
                progress_callback=_phase_cb,
                replay_level=replay_level,
                ms_rounds_per_ec_round=ms_rounds_per_ec_round or 4,
                max_sat_time=max_sat_time,
                max_rc2_time=max_rc2_time,
                solver_params=solver_params,
            )
            phase_steps = list(phase_steps_raw)

        # --- Global ms_round_index renumbering ---
        # FIX 21: Use non-mutating replace so cached step objects keep
        # their LOCAL ms_round_index. Mutating in-place caused
        # double-renumbering on cache replay → non-monotonic indices
        # → cascading PRE-FLIGHT FAILs.
        phase_steps = [
            _dc_replace(step, ms_round_index=step.ms_round_index + round_cursor)
            for step in phase_steps
        ]
        if round_cursor == 0 and phase_steps:
            phase_steps[0] = _dc_replace(phase_steps[0], is_initial_placement=True)

        all_routing_steps.extend(phase_steps)

        # ── Fix E: blocks_pure diagnostic after each phase ──
        if ion_to_block and block_sub_grids and phase_steps:
            _post_layout = phase_steps[-1].layout_after
            _misplaced: Dict[str, List[int]] = {}
            for _bname, _sg in block_sub_grids.items():
                _r0, _c0, _r1, _c1 = _sg.grid_region
                _c0i, _c1i = _c0 * k, _c1 * k
                _blk_slice = _post_layout[_r0:_r1, _c0i:_c1i]
                for _ri in range(_blk_slice.shape[0]):
                    for _ci in range(_blk_slice.shape[1]):
                        _ion = int(_blk_slice[_ri, _ci])
                        if _ion != 0 and _ion in ion_to_block and ion_to_block[_ion] != _bname:
                            _misplaced.setdefault(_bname, []).append(_ion)
            _pt_label = getattr(plan.phase_type, 'name', plan.phase_type)
            if _misplaced:
                raise ValueError(
                    f"[BlockPurity] Phase {phase_idx + 1}/{len(plans)} "
                    f"({_pt_label}): IMPURE after routing — ions in wrong "
                    f"blocks at phase exit: "
                    f"{({b: sorted(ions) for b, ions in _misplaced.items()})}. "
                    f"The merge or route-back produced a layout with ions "
                    f"outside their home sub-grids."
                )
            else:
                logger.info(
                    "[BlockPurity] Phase %d/%d (%s): PURE — all ions in home blocks",
                    phase_idx + 1, len(plans), _pt_label,
                )

        # Advance round_cursor past ALL indices used in this phase,
        # including post-gadget transition steps (which sit at n_pairs,
        # beyond the return-round capping at n_pairs - 1).  Without
        # this, the next phase's first step (index 0 + new round_cursor)
        # collides with this phase's transition steps.
        _max_local_idx = max(
            (s.ms_round_index - round_cursor for s in phase_steps),
            default=n_pairs - 1,
        )
        _phase_span = max(n_pairs, _max_local_idx + 1)
        round_cursor += _phase_span
        _global_round_offset += _phase_span

    # ── Fix 20: Sort steps by ms_round_index BEFORE ChainVerify ──
    # The execution loop (ionRoutingWISEArch) sorts routing_steps by
    # ms_round_index and processes them in that order via groupby.
    # ChainVerify must verify steps in the SAME order, otherwise its
    # chain layout diverges from execution's oldArrangementArr.
    # Without this sort, steps produced by different tiling passes
    # within the same ms_round may appear in a different order than
    # execution expects, causing cascading PRE-FLIGHT FAILs.
    all_routing_steps = sorted(
        all_routing_steps, key=lambda s: s.ms_round_index,
    )

    # ── Fix 16: Post-planning chain verification ─────────────────
    # Walk the step chain with the initial layout, verify that each
    # step's schedule transforms the current layout to layout_after.
    # Rebuild broken schedules DURING PLANNING so the execution loop
    # never needs to invoke SAT.
    #
    # Catches:
    #   - Schedules dropped by ion conservation repair in merge
    #   - Stale schedules from cache replay on drifted layouts
    #   - Missing layout_before on transition/return steps
    #   - Chain continuity breaks at phase boundaries
    _chain_layout = np.array(initial_layout, copy=True)
    _chain_rebuilds = 0
    _chain_skipped = 0
    _chain_continuity_fixes = 0
    _chain_verified: List[RoutingStep] = []

    for _si, _step in enumerate(all_routing_steps):
        # 1. Fix layout_before
        if _step.layout_before is None:
            _step.layout_before = np.array(_chain_layout, copy=True)
        elif (
            _step.layout_before.shape == _chain_layout.shape
            and not np.array_equal(_step.layout_before, _chain_layout)
        ):
            _lb_diff = int(np.sum(_step.layout_before != _chain_layout))
            logger.debug(
                "[ChainVerify] step %d (ms_round=%d): layout_before "
                "differs from chain by %d cells — updating",
                _si, _step.ms_round_index, _lb_diff,
            )
            _step.layout_before = np.array(_chain_layout, copy=True)
            _chain_continuity_fixes += 1

        # 2. No-op: layouts already match — no reconfig needed
        if np.array_equal(_chain_layout, _step.layout_after):
            _chain_verified.append(_step)
            continue

        # 3. Verify schedule produces the target layout
        _cv_need_rebuild = _step.schedule is None
        if not _cv_need_rebuild:
            _cv_replay = _simulate_schedule_replay(
                _chain_layout, _step.schedule,
            )
            _cv_need_rebuild = not np.array_equal(
                _cv_replay, _step.layout_after,
            )

        if _cv_need_rebuild:
            # ══════════════════════════════════════════════════════════
            # ChainVerify HEURISTIC POLICY — DO NOT MODIFY / REVERT
            # ══════════════════════════════════════════════════════════
            # This policy determines whether _rebuild_schedule_for_layout
            # may fall back to heuristic odd-even transposition sort or
            # must use SAT-only routing.
            #
            # - ms_gate context:  allow_heuristic = False
            #     MS-gate reconfigs MUST use SAT because the schedule
            #     encodes physical ion movements that affect gate
            #     fidelity.  Heuristic would produce a valid layout
            #     but with a non-optimised schedule.
            #
            # - non-MS contexts (cache_replay, return_round, transition):
            #     allow_heuristic = True
            #     These are pure layout permutations where the odd-even
            #     sort heuristic always produces the exact target layout.
            #     Using SAT here adds 20s+ overhead per rebuild with no
            #     fidelity benefit.
            #
            # If you change this policy, the compiler will either:
            #   - Stall indefinitely (if you set ms_gate to True)
            #   - Produce sub-optimal gate schedules (if you set
            #     non-MS to False when the user has disabled heuristics)
            # ══════════════════════════════════════════════════════════
            # 4. Rebuild schedule during planning
            if _step.reconfig_context == "ms_gate":
                _cv_allow_heuristic = False
            else:
                # Non-MS contexts (cache_replay, return_round, transitions)
                # are pure layout permutations where the odd-even sort
                # heuristic always produces the exact target.  Always
                # allow heuristic fallback to prevent unbounded SAT stalls.
                _cv_allow_heuristic = True
            _cv_diff = int(
                np.sum(_chain_layout != _step.layout_after)
            )
            logger.info(
                "[ChainVerify] step %d (ms_round=%d, ctx=%s): "
                "rebuilding schedule (%s, %d/%d cells to target)",
                _si, _step.ms_round_index,
                _step.reconfig_context,
                "schedule=None" if _step.schedule is None
                else "replay mismatch",
                _cv_diff, _chain_layout.size,
            )

            if _cv_allow_heuristic:
                # ── FAST PATH: non-MS contexts ──────────────────
                # Skip SAT entirely.  Emit a single heuristic
                # snapshot (schedule=None).  physicalOperation will
                # use the deterministic odd-even transposition sort
                # which always produces the exact target layout.
                # This eliminates 20s+ SAT overhead per rebuild.
                _cv_snaps = [(_step.layout_after.copy(), None, [])]
                logger.info(
                    "[ChainVerify] step %d: using heuristic directly "
                    "(non-MS context '%s', %d cells to target)",
                    _si, _step.reconfig_context, _cv_diff,
                )
            else:
                try:
                    _cv_snaps = _rebuild_schedule_for_layout(
                        _chain_layout.copy(), wiseArch,
                        _step.layout_after,
                        subgridsize=subgridsize,
                        base_pmax_in=base_pmax_in or 1,
                        stop_event=stop_event,
                        max_inner_workers=max_inner_workers,
                        allow_heuristic_fallback=_cv_allow_heuristic,
                        max_sat_time=max_sat_time,
                        max_rc2_time=max_rc2_time,
                        solver_params=solver_params,
                    )
                except Exception as _cv_exc:
                    logger.warning(
                        "[ChainVerify] step %d: rebuild failed: %s",
                        _si, _cv_exc,
                    )
                    _cv_snaps = []

            if _cv_snaps:
                # Insert intermediate SAT cycles as extra steps
                for _cv_lay, _cv_sched, _ in _cv_snaps[:-1]:
                    _chain_verified.append(RoutingStep(
                        layout_after=np.array(_cv_lay, copy=True),
                        schedule=_cv_sched,
                        solved_pairs=[],
                        ms_round_index=_step.ms_round_index,
                        from_cache=_step.from_cache,
                        tiling_meta=_step.tiling_meta,
                        can_merge_with_next=True,
                        is_initial_placement=False,
                        is_layout_transition=True,
                        layout_before=np.array(
                            _chain_layout, copy=True,
                        ),
                        reconfig_context=_step.reconfig_context,
                    ))
                    _chain_layout = np.array(_cv_lay, copy=True)

                # Use last cycle's schedule for this step
                _cv_last_lay, _cv_last_sched, _ = _cv_snaps[-1]
                _step.schedule = _cv_last_sched
                _step.layout_before = np.array(
                    _chain_layout, copy=True,
                )
                # Verify last schedule matches target
                if _cv_last_sched is not None:
                    _cv_final = _simulate_schedule_replay(
                        _chain_layout, _cv_last_sched,
                    )
                    if not np.array_equal(
                        _cv_final, _step.layout_after,
                    ):
                        # Use SAT's actual result layout
                        _step.layout_after = np.array(
                            _cv_last_lay, copy=True,
                        )
                _chain_rebuilds += 1
                logger.info(
                    "[ChainVerify] step %d: rebuilt via %d SAT "
                    "cycle(s)",
                    _si, len(_cv_snaps),
                )
            else:
                # Rebuild failed — leave as-is for execution fallback
                _chain_skipped += 1
                logger.warning(
                    "[ChainVerify] step %d: rebuild returned 0 "
                    "snapshots — deferring to execution fallback",
                    _si,
                )

        _chain_verified.append(_step)
        _chain_layout = np.array(_step.layout_after, copy=True)

    if _chain_rebuilds > 0 or _chain_skipped > 0:
        logger.info(
            "[ChainVerify] rebuilt %d schedules, %d deferred to "
            "execution, %d continuity fixes; steps: %d → %d",
            _chain_rebuilds, _chain_skipped,
            _chain_continuity_fixes,
            len(all_routing_steps), len(_chain_verified),
        )
    else:
        logger.info(
            "[ChainVerify] all %d steps verified — no rebuilds needed",
            len(all_routing_steps),
        )
    all_routing_steps = _chain_verified

    logger.info(
        "[PhaseSteps] total %d routing steps across %d phases",
        len(all_routing_steps), len(plans),
    )

    # Emit final STAGE_COMPLETE so the progress bar shows 100%
    if progress_callback is not None:
        progress_callback(RoutingProgress(
            stage=STAGE_COMPLETE,
            current=_total_global_rounds,
            total=_total_global_rounds,
            message=(
                f"Routing complete: {len(all_routing_steps)} steps, "
                f"{len(plans)} phases, "
                f"{_reconfig_count} reconfigurations"
            ),
        ))

    return all_routing_steps, current_layout