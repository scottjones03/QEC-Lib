# src/qectostim/experiments/hardware_simulation/trapped_ion/clustering.py
"""
Heuristic ion-to-trap clustering and placement.

Ports the clustering / qubit-to-ion partitioning algorithms from the old
``compiler/qccd_qubits_to_ions.py`` into the new architecture layer:

* **regularPartition** — BSP-style recursive median split that groups
  measurement + data ions into clusters fitting a trap capacity.
* **hillClimbOnArrangeClusters** — Hungarian-matching based cluster
  placement with hill-climbing optimisation.

These are the *non-WISE, non-SAT* qubit mapping strategies used for the
Augmented Grid and Networked Grid architectures.

References
----------
* Original implementation:
  ``old/src/compiler/qccd_qubits_to_ions.py``
"""

from __future__ import annotations

import itertools
from typing import (
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
)

import numpy as np
import numpy.typing as npt
from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance_matrix

from qectostim.experiments.hardware_simulation.trapped_ion.architecture import Ion

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAX_ITER: int = 20_000

# Type alias: a cluster is (list-of-ions, centre-coordinate)
Cluster = Tuple[List[Ion], npt.NDArray[np.float64]]


# ============================================================================
# Cluster merging
# ============================================================================

def merge_clusters_to_limit(
    clusters: Sequence[Cluster],
    max_clusters: int,
    capacity: int,
) -> List[Cluster]:
    """Greedily merge clusters until ``len(clusters) <= max_clusters``.

    Never exceeds *capacity* ions per cluster.

    Strategy
    --------
    1. Repeatedly merge the closest pair of clusters whose combined
       size ≤ *capacity*, updating the centre as a weighted mean.
    2. If no such pair exists but we still have too many clusters,
       attempt to *dissolve* a small cluster by redistributing its ions
       into other clusters that still have free capacity.
    3. If even dissolution is impossible, raise ``RuntimeError``.

    Parameters
    ----------
    clusters : sequence of (ions, centre)
        Input clusters.
    max_clusters : int
        Maximum number of output clusters.
    capacity : int
        Maximum number of ions per cluster.

    Returns
    -------
    list of (ions, centre)
    """
    # Mutable copy
    clusters_mut: List[Cluster] = [
        (list(ions), np.array(centre, dtype=float))
        for ions, centre in clusters
    ]

    total_ions = sum(len(ions) for ions, _ in clusters_mut)
    if max_clusters * capacity < total_ions:
        raise RuntimeError(
            f"Insufficient total capacity: max_clusters={max_clusters}, "
            f"capacity={capacity}, total_ions={total_ions}"
        )

    def _total_free() -> int:
        return sum(max(0, capacity - len(ions)) for ions, _ in clusters_mut)

    while len(clusters_mut) > max_clusters:
        # ---- 1. Try merging closest pair ----
        best_pair = None
        best_dist2: float = float("inf")

        for i in range(len(clusters_mut)):
            ions_i, centre_i = clusters_mut[i]
            for j in range(i + 1, len(clusters_mut)):
                ions_j, centre_j = clusters_mut[j]
                if len(ions_i) + len(ions_j) > capacity:
                    continue
                d2 = float(np.sum((centre_i - centre_j) ** 2))
                if d2 < best_dist2:
                    best_pair = (i, j)
                    best_dist2 = d2

        if best_pair is not None:
            i, j = best_pair
            ions_i, centre_i = clusters_mut[i]
            ions_j, centre_j = clusters_mut[j]

            merged_ions = ions_i + ions_j
            merged_centre = (
                centre_i * len(ions_i) + centre_j * len(ions_j)
            ) / float(len(merged_ions))

            for idx in sorted((i, j), reverse=True):
                clusters_mut.pop(idx)
            clusters_mut.append((merged_ions, merged_centre))
            continue

        # ---- 2. Try dissolving a small cluster ----
        dissolved = False
        donor_candidates = sorted(
            enumerate(clusters_mut), key=lambda x: len(x[1][0])
        )

        for donor_idx, (donor_ions, donor_centre) in donor_candidates:
            if not donor_ions:
                clusters_mut.pop(donor_idx)
                dissolved = True
                break

            receivers = []
            for j, (ions_j, centre_j) in enumerate(clusters_mut):
                if j == donor_idx:
                    continue
                free = capacity - len(ions_j)
                if free > 0:
                    receivers.append((j, free, centre_j))

            if not receivers:
                continue

            total_free = sum(f for _, f, _ in receivers)
            if total_free < len(donor_ions):
                continue

            receivers.sort(
                key=lambda t: float(np.sum((donor_centre - t[2]) ** 2))
            )

            ions_to_assign = list(donor_ions)
            for j, free, _ in receivers:
                if not ions_to_assign:
                    break
                take = min(free, len(ions_to_assign))
                moved = ions_to_assign[:take]
                ions_to_assign = ions_to_assign[take:]

                ions_j, centre_j_curr = clusters_mut[j]
                old_size = len(ions_j)
                ions_j.extend(moved)
                new_centre = (
                    centre_j_curr * old_size + donor_centre * len(moved)
                ) / float(old_size + len(moved))
                clusters_mut[j] = (ions_j, new_centre)

            if ions_to_assign:
                raise RuntimeError(
                    "Internal error in merge_clusters_to_limit: "
                    "failed to reassign all donor ions."
                )

            clusters_mut.pop(donor_idx)
            dissolved = True
            break

        if dissolved:
            continue

        # ---- 3. Impossible ----
        raise RuntimeError(
            f"Cannot reduce clusters to max_clusters={max_clusters} "
            f"without exceeding capacity={capacity}. "
            f"#clusters={len(clusters_mut)}, total_ions={total_ions}, "
            f"total_free_capacity={_total_free()}."
        )

    return clusters_mut


# ============================================================================
# BSP recursive-median partitioning
# ============================================================================

def _partition_cluster_ions(
    ions: Sequence[Ion],
    coords: npt.NDArray[np.float64],
    trap_capacity: int,
) -> List[Cluster]:
    """Recursive median-split BSP partition of ions into clusters.

    Alternates splitting on X and Y axes until every cluster fits in
    *trap_capacity*.

    Parameters
    ----------
    ions : sequence of Ion
    coords : ndarray of shape (N, 2)
    trap_capacity : int

    Returns
    -------
    list of (ions, centre)
    """
    # Track original indices through the partition so we can map back to ions.
    indexed: List[Tuple[int, npt.NDArray[np.float64]]] = [
        (i, coords[i]) for i in range(len(coords))
    ]
    partitions: List[List[Tuple[int, npt.NDArray[np.float64]]]] = [indexed]
    split_axis_is_x = True

    _max_splits = 200  # safety: prevent infinite loop on degenerate coords
    _splits = 0
    while max(len(p) for p in partitions) > trap_capacity:
        _splits += 1
        if _splits > _max_splits:
            break
        to_split = [p for p in partitions if len(p) > trap_capacity]
        for p in to_split:
            axis_vals = [float(item[1][int(split_axis_is_x)]) for item in p]
            median = float(np.mean(axis_vals))
            p1, p2 = [], []
            for item, v in zip(p, axis_vals):
                (p1 if v <= median else p2).append(item)
            # If all coords are identical the split is degenerate —
            # force an arbitrary 50/50 split so we always shrink.
            if not p2:
                half = trap_capacity
                p1, p2 = p[:half], p[half:]
            if p1:
                partitions.append(p1)
            if p2:
                partitions.append(p2)
        split_axis_is_x = not split_axis_is_x
        for p in to_split:
            partitions.remove(p)

    clusters: List[Cluster] = []
    for p in partitions:
        cluster_ions = [ions[idx] for idx, _ in p]
        centre = np.mean([c for _, c in p], axis=0)
        clusters.append((cluster_ions, centre))
    return clusters


# ============================================================================
# Main partition entry point
# ============================================================================

def regular_partition(
    measurement_ions: Sequence[Ion],
    data_ions: Sequence[Ion],
    trap_capacity: int,
    *,
    is_wise_arch: bool = False,
    max_clusters: Optional[int] = None,
) -> List[Cluster]:
    """Partition measurement and data ions into clusters.

    Each cluster fits within *trap_capacity* (or *trap_capacity − 1*
    when ``is_wise_arch`` is False, reserving one slot for routing).

    Optionally enforces a maximum number of clusters by merging
    nearby clusters.

    Parameters
    ----------
    measurement_ions, data_ions : sequence of Ion
    trap_capacity : int
    is_wise_arch : bool
        If True, full capacity is used (WISE does not need the extra
        slot for routing).
    max_clusters : int, optional
        Merge down to this many clusters if set.

    Returns
    -------
    list of (ions, centre)
    """
    eff_capacity = trap_capacity - (0 if is_wise_arch else 1)
    d_ions_per_trap = trap_capacity

    while True:
        m_ions_l = list(measurement_ions)
        m_coords = np.array([list(ion.position) for ion in m_ions_l])

        d_ions_l = list(data_ions)
        d_coords = np.array([list(ion.position) for ion in d_ions_l])

        clusters_d = list(_partition_cluster_ions(d_ions_l, d_coords, d_ions_per_trap))
        clusters_m = list(_partition_cluster_ions(m_ions_l, m_coords, 1))

        clusters = list(clusters_d)

        # Attach each measurement cluster to the nearest data cluster
        for cluster_m in clusters_m:
            nearest = min(
                clusters,
                key=lambda c: (
                    (c[1][0] - cluster_m[1][0]) ** 2
                    + (c[1][1] - cluster_m[1][1]) ** 2
                ),
            )
            merged_ions = list(nearest[0]) + list(cluster_m[0])
            ratio_d = len(nearest[0]) / len(merged_ions)
            new_centre = cluster_m[1] * (1 - ratio_d) + nearest[1] * ratio_d
            clusters.append((merged_ions, new_centre))
            clusters.remove(nearest)

        max_size = max(len(c[0]) for c in clusters)

        if max_size > eff_capacity:
            if d_ions_per_trap == 2:
                # Fallback: cluster all ions together
                all_ions = list(measurement_ions) + list(data_ions)
                all_coords = np.array([list(ion.position) for ion in all_ions])
                clusters = _partition_cluster_ions(all_ions, all_coords, eff_capacity)
                break
            d_ions_per_trap -= 1
        else:
            break

    if max_clusters is not None and len(clusters) > max_clusters:
        # Use full trap_capacity for merging — the routing-slot reservation
        # (eff_capacity) is for operational scheduling, not initial placement.
        clusters = merge_clusters_to_limit(clusters, max_clusters, trap_capacity)

    return clusters


# ============================================================================
# Hungarian matching with variance-enhanced cost
# ============================================================================

def _min_weight_perfect_match(
    A: npt.NDArray[np.float64],
    B_subset: npt.NDArray[np.float64],
    centralizer: npt.NDArray[np.float64],
    divider: npt.NDArray[np.float64],
    nearest_coords_A: npt.NDArray[np.float64],
    nearest_dists_A: npt.NDArray[np.float64],
) -> Tuple[float, npt.NDArray[np.intp]]:
    """Variance-enhanced Hungarian matching.

    Normalises *B_subset* and combines Euclidean distance with a
    nearest-neighbour variance term for a more topology-preserving
    assignment.

    Returns
    -------
    (total_cost, column_indices)
    """
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
        cost_matrix = distance_matrix(A, rel_B) + variance_matrix
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
    except ValueError:
        cost_matrix = distance_matrix(A, rel_B)
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
    total_cost = float(cost_matrix[row_ind, col_ind].sum())
    return total_cost, col_ind


# ============================================================================
# Cluster arrangement on a grid (single biasY)
# ============================================================================

def _arrange_clusters(
    clusters: Sequence[Cluster],
    all_grid_pos: Sequence[Tuple[int, int]],
    nearest_neighbour_count: int = 4,
    bias_y: int = 1,
) -> Tuple[List[Tuple[int, int]], float]:
    """Place clusters on grid positions via Hungarian matching.

    Parameters
    ----------
    clusters : sequence of (ions, centre)
    all_grid_pos : available (col, row) positions
    nearest_neighbour_count : int
        Neighbours used for variance cost.
    bias_y : int
        Vertical bias factor for centroid selection.

    Returns
    -------
    (grid_positions, cost)
        ``grid_positions[i]`` is the (col, row) assigned to ``clusters[i]``.
    """
    A = np.array([c[1] for c in clusters])
    if len(A) == 0 or len(all_grid_pos) == 0:
        return [], float("inf")

    min_x, min_y = float(A[:, 0].min()), float(A[:, 1].min())
    max_x, max_y = float(A[:, 0].max()), float(A[:, 1].max())
    dx = max_x - min_x or 1.0
    dy = max_y - min_y or 1.0

    centralizer = np.tile([min_x, min_y], (len(A), 1))
    divider = np.tile([dx, dy], (len(A), 1))
    A_norm = (A - centralizer) / divider

    dist_A = distance_matrix(A_norm, A_norm)
    np.fill_diagonal(dist_A, np.inf)
    nn_count = min(nearest_neighbour_count, len(A) - 1) if len(A) > 1 else 0
    if nn_count > 0:
        nn_indices = np.argsort(dist_A, axis=1)[:, :nn_count]
        nearest_coords_A = A_norm[nn_indices]
        nearest_dists_A = dist_A[np.arange(len(A))[:, None], nn_indices]
    else:
        nearest_coords_A = np.empty((len(A), 0, 2))
        nearest_dists_A = np.empty((len(A), 0))

    cardinality_A = len(A)
    if cardinality_A > len(all_grid_pos):
        raise ValueError("Not enough traps for the number of clusters")

    # --- Build candidate centroid set ---
    centroid_B = np.mean(all_grid_pos, axis=0)
    sorted_to_centroid = sorted(
        all_grid_pos,
        key=lambda p: (p[0] - centroid_B[0]) ** 2 + (p[1] - centroid_B[1]) ** 2,
    )

    around_centroid: List[Tuple[int, int]] = []
    for xsign, ysign in [
        (-1, 0), (1, 0), (0, 1), (0, -1), (0, 0),
        (1, 1), (-1, -1), (1, -1), (-1, 1),
    ]:
        for p in sorted_to_centroid:
            match_x = (
                (xsign == -1 and p[0] < centroid_B[0])
                or (xsign == 1 and p[0] > centroid_B[0])
                or xsign == 0
            )
            match_y = (
                (ysign == -1 and p[1] < centroid_B[1])
                or (ysign == 1 and p[1] > centroid_B[1])
                or ysign == 0
            )
            if match_x and match_y:
                around_centroid.append(p)
                break

    best_cost = float("inf")
    best_map: List[Tuple[int, int]] = []

    for centroid_B_cand in set(around_centroid):
        not_picked: Dict[float, List[Tuple[int, int]]] = {}
        for p in all_grid_pos:
            dis = max(
                ((p[0] - centroid_B_cand[0]) ** 2) * bias_y,
                (p[1] - centroid_B_cand[1]) ** 2,
            )
            not_picked.setdefault(dis, []).append(p)

        guaranteed = []
        next_window = []
        while not_picked:
            next_window = not_picked.pop(min(not_picked.keys()))
            if len(guaranteed) + len(next_window) < cardinality_A:
                guaranteed.extend(next_window)
            else:
                break

        if not next_window:
            continue

        wmin_x = min(w[0] for w in next_window)
        wmax_x = max(w[0] for w in next_window)
        wmin_y = min(w[1] for w in next_window)
        wmax_y = max(w[1] for w in next_window)
        wdx = wmax_x - wmin_x or 1
        wdy = wmax_y - wmin_y or 1
        cent_mat = np.tile([wmin_x, wmin_y], (cardinality_A, 1)).astype(float)
        div_mat = np.tile([wdx, wdy], (cardinality_A, 1)).astype(float)

        # Build sorted boundary window
        bottom = sorted([p for p in next_window if p[1] == wmin_y], key=lambda p: p[0])
        right = sorted([p for p in next_window if p[0] == wmax_x], key=lambda p: p[1])
        top = sorted([p for p in next_window if p[1] == wmax_y], key=lambda p: p[0], reverse=True)
        left = sorted([p for p in next_window if p[0] == wmin_x], key=lambda p: p[1], reverse=True)
        sorted_window = bottom[:-1] + right[:-1] + top[:-1] + left[:-1]
        if not sorted_window:
            sorted_window = bottom

        needed = cardinality_A - len(guaranteed)
        reg_spacing = int(len(sorted_window) / needed) if needed > 0 else 0
        if reg_spacing == 0:
            continue

        in_B = [sorted_window[i * reg_spacing] for i in range(needed)]
        not_in_B = [w for w in sorted_window if w not in in_B]
        B_subset = np.array(in_B + guaranteed, dtype=float)

        total_cost, _ = _min_weight_perfect_match(
            A_norm, B_subset, cent_mat, div_mat, nearest_coords_A, nearest_dists_A
        )
        current_score = total_cost

        # --- Hill-climbing swap loop ---
        _it = 0
        while True:
            next_in_B = []
            next_score = float("inf")
            for i in range(len(in_B)):
                for b_not in not_in_B:
                    cand = in_B[:i] + [b_not] + in_B[i + 1:]
                    B_cand = np.array(cand + guaranteed, dtype=float)
                    tc, _ = _min_weight_perfect_match(
                        A_norm, B_cand, cent_mat, div_mat,
                        nearest_coords_A, nearest_dists_A,
                    )
                    if tc < next_score:
                        next_in_B = cand
                        next_score = tc
                    if tc == 0:
                        break

            if current_score <= next_score:
                break
            if _it > MAX_ITER:
                break

            in_B = next_in_B
            not_in_B = [w for w in sorted_window if w not in in_B]
            current_score = next_score
            if current_score == 0:
                break
            _it += 1

        if current_score < best_cost:
            B_subset = np.array(in_B + guaranteed, dtype=float)
            _, col_ind = _min_weight_perfect_match(
                A_norm, B_subset, cent_mat, div_mat,
                nearest_coords_A, nearest_dists_A,
            )
            best_map = [tuple(B_subset[idx].astype(int)) for idx in col_ind]
            best_cost = current_score
        if best_cost == 0:
            break

    return best_map, best_cost


def arrange_clusters(
    clusters: Sequence[Cluster],
    all_grid_pos: Sequence[Tuple[int, int]],
    nearest_neighbour_count: int = 4,
    bias_y: int = 1,
) -> List[Tuple[int, int]]:
    """Place clusters on grid positions (returns positions only)."""
    positions, _ = _arrange_clusters(clusters, all_grid_pos, nearest_neighbour_count, bias_y)
    return positions


# ============================================================================
# Hill-climbing over biasY
# ============================================================================

def hill_climb_on_arrange_clusters(
    clusters: Sequence[Cluster],
    all_grid_pos: Sequence[Tuple[int, int]],
    nearest_neighbour_count: int = 4,
) -> List[Tuple[int, int]]:
    """Optimise cluster placement by hill-climbing over *biasY*.

    Iterates over increasing ``bias_y`` values, keeping the best
    placement found.

    Parameters
    ----------
    clusters : sequence of (ions, centre)
    all_grid_pos : available (col, row) positions
    nearest_neighbour_count : int

    Returns
    -------
    list of (col, row) — one per cluster, in the same order.
    """
    bias_y = 1
    current_cost = float("inf")
    current_map: List[Tuple[int, int]] = []
    next_cost = float("inf")
    next_map: List[Tuple[int, int]] = []

    iters = 0
    while True:
        current_map, current_cost = next_map, next_cost
        next_cost = float("inf")
        next_map = []

        best_i = 0
        for i in range(10):
            _map, _cost = _arrange_clusters(
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
            return current_map
        if iters > MAX_ITER:
            return current_map
        bias_y += 10

    return current_map  # pragma: no cover – unreachable


__all__ = [
    "Cluster",
    "merge_clusters_to_limit",
    "regular_partition",
    "arrange_clusters",
    "hill_climb_on_arrange_clusters",
]
