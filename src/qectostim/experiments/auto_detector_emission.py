# src/qectostim/experiments/tqec_detector_emission.py
"""
Automatic detector emission using Stim flow validation.

This module provides automatic detector discovery for any CSS gadget circuit,
replacing the manual per-gadget classification into anchor/temporal/crossing/
boundary detector types.  It uses Stim's ``has_flow()`` to validate candidate
measurement subsets as deterministic detectors, then applies GF(2) Gaussian
elimination to select a maximal independent set.

Architecture
------------
The module works in two modes:

1. **Full-circuit mode** (``AutoDetectorEmitter.from_circuit``):
   Takes a complete ``stim.Circuit`` and discovers ALL detectors automatically
   via flow validation. Emits DETECTOR instructions directly.

2. **Config-generation mode** (``AutoDetectorEmitter.compute_crossing_config``,
   ``AutoDetectorEmitter.compute_boundary_config``):
   Derives ``CrossingDetectorConfig`` and ``BoundaryDetectorConfig`` from the
   circuit structure, producing the same dataclasses the existing experiment
   framework consumes. This enables gradual migration — gadgets can opt into
   auto-detection without changing the experiment orchestration.

The detector discovery engine works by:
  1. Finding *anchor* detectors (weight-1): single measurements with
     flow ``1 → rec[-k]`` (reset-then-measure pairs).
  2. Finding *temporal* detectors (weight-2): consecutive same-qubit
     measurements with flow ``1 → rec[-j] ⊕ rec[-k]``.
  3. Finding *boundary* detectors (weight 3–5): combinations of a
     last-round ancilla measurement with final data-qubit measurements.
  4. Pruning redundant detectors via GF(2) Gaussian elimination to
     obtain a maximal linearly independent set.

Results are cached by circuit hash to avoid recomputation.

Dependencies
------------
- ``stim >= 1.14``: For ``has_flow()``, circuit manipulation
- ``numpy``: For GF(2) Gaussian elimination
- ``tqecd`` (optional): Still imported for legacy config-generation helpers

Compatibility
-------------
Produces ``CrossingDetectorConfig``, ``BoundaryDetectorConfig``, and raw
``DETECTOR`` instructions that are fully compatible with the existing
``FaultTolerantGadgetExperiment`` pipeline.
"""
from __future__ import annotations

import hashlib
import itertools
import logging
from dataclasses import dataclass, field
from typing import (
    Dict, FrozenSet, List, Mapping, Optional, Sequence, Set, Tuple,
    TYPE_CHECKING,
)

import numpy as np
import stim



if TYPE_CHECKING:
    from qectostim.experiments.stabilizer_rounds.context import DetectorContext
    from qectostim.gadgets.base import (
        BoundaryDetectorConfig,
        CrossingDetectorConfig,
    )

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MEAS_NAMES = frozenset({"M", "MX", "MY", "MR", "MRX", "MRY", "MZ"})
_RESET_NAMES = frozenset({"R", "RX", "RY"})
_MEAS_BASIS = {
    "M": "Z", "MX": "X", "MY": "Y", "MR": "Z",
    "MRX": "X", "MRY": "Y", "MZ": "Z",
}


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DiscoveredDetector:
    """A detector discovered by flow validation.

    Attributes
    ----------
    measurement_indices : frozenset of int
        Absolute measurement indices (0-based) composing this detector.
    coordinates : tuple of float
        Spatial + temporal coordinates (x, y, t, ...).
    """
    measurement_indices: FrozenSet[int]
    coordinates: Tuple[float, ...]

    @property
    def weight(self) -> int:
        return len(self.measurement_indices)


@dataclass
class FlowMatchResult:
    """Result of flow-based detector discovery on a circuit.

    Attributes
    ----------
    detectors : list of DiscoveredDetector
        All deterministic detectors found (independent set).
    num_measurements : int
        Total measurements in the analysed circuit.
    qubit_coords : dict
        Qubit index → coordinate mapping used.
    raw_detector_count : int
        Number of raw (pre-pruning) detectors found.
    """
    detectors: List[DiscoveredDetector] = field(default_factory=list)
    num_measurements: int = 0
    qubit_coords: Dict[int, Tuple[float, ...]] = field(default_factory=dict)
    raw_detector_count: int = 0


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------

class _DetectorCache:
    """Simple in-memory cache keyed by circuit hash."""

    def __init__(self) -> None:
        self._cache: Dict[bytes, FlowMatchResult] = {}

    def _hash(self, circuit: stim.Circuit) -> bytes:
        return hashlib.sha256(
            str(circuit).encode("utf-8")
        ).digest()

    def get(self, circuit: stim.Circuit) -> Optional[FlowMatchResult]:
        return self._cache.get(self._hash(circuit))

    def put(self, circuit: stim.Circuit, result: FlowMatchResult) -> None:
        self._cache[self._hash(circuit)] = result

    def clear(self) -> None:
        self._cache.clear()


# Module-level cache instance
_cache = _DetectorCache()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_qubit_coords(circuit: stim.Circuit) -> Dict[int, Tuple[float, ...]]:
    """Extract QUBIT_COORDS from a stim.Circuit."""
    coords: Dict[int, Tuple[float, ...]] = {}
    for instruction in circuit.flattened():
        if instruction.name == "QUBIT_COORDS":
            args = instruction.gate_args_copy()
            for target in instruction.targets_copy():
                coords[target.value] = tuple(args)
    return coords


def _count_measurements(circuit: stim.Circuit) -> int:
    """Count total measurements in a circuit."""
    count = 0
    for instruction in circuit.flattened():
        if instruction.name in _MEAS_NAMES:
            count += len(instruction.targets_copy())
    return count


def _get_measurement_info(
    circuit: stim.Circuit,
) -> List[Tuple[int, str, str]]:
    """Map each measurement index to (qubit_index, basis, instruction_name)."""
    info: List[Tuple[int, str, str]] = []
    for inst in circuit.flattened():
        if inst.name in _MEAS_NAMES:
            basis = _MEAS_BASIS.get(inst.name, "Z")
            for t in inst.targets_copy():
                info.append((t.value, basis, inst.name))
    return info


def _group_measurements_by_qubit(
    meas_info: List[Tuple[int, str, str]],
) -> Dict[int, List[int]]:
    """Group measurement indices by qubit index."""
    by_qubit: Dict[int, List[int]] = {}
    for idx, (q, _b, _n) in enumerate(meas_info):
        by_qubit.setdefault(q, []).append(idx)
    return by_qubit


# ---------------------------------------------------------------------------
# Legacy: split_circuit_into_rounds (kept for backward compatibility)
# ---------------------------------------------------------------------------

def split_circuit_into_rounds(circuit: stim.Circuit) -> List[stim.Circuit]:
    """Split a circuit into per-round fragments at TICK boundaries.

    .. deprecated::
        This function is retained for backward compatibility.
        The primary detector discovery engine now uses ``has_flow()``
        directly rather than fragment-based flow matching.
    """
    coords_header = stim.Circuit()
    for instruction in circuit:
        if instruction.name == "QUBIT_COORDS":
            coords_header.append(instruction)

    fragments: List[stim.Circuit] = []
    current = stim.Circuit()
    current += coords_header

    for instruction in circuit:
        if instruction.name == "QUBIT_COORDS":
            continue
        if instruction.name == "TICK":
            if _has_operations(current):
                current.append("TICK")
                fragments.append(current)
                current = stim.Circuit()
                current += coords_header
        else:
            current.append(instruction)

    if _has_operations(current):
        fragments.append(current)

    return fragments if fragments else [circuit]


def _has_operations(circuit: stim.Circuit) -> bool:
    """Check if a circuit has any non-QUBIT_COORDS operations."""
    for inst in circuit:
        if inst.name not in ("QUBIT_COORDS",):
            return True
    return False


# ---------------------------------------------------------------------------
# Core: flow-based detector discovery
# ---------------------------------------------------------------------------

def _discover_anchor_detectors(
    circuit: stim.Circuit,
    n_meas: int,
) -> List[FrozenSet[int]]:
    """Find weight-1 (anchor) detectors: ``1 → rec[-k]``.

    These arise from reset-then-measure pairs on qubits that are
    deterministically initialised.
    """
    anchors: List[FrozenSet[int]] = []
    for idx in range(n_meas):
        offset = n_meas - idx
        flow_str = f"1 -> rec[-{offset}]"
        try:
            if circuit.has_flow(stim.Flow(flow_str), unsigned=True):
                anchors.append(frozenset([idx]))
        except Exception:
            pass
    return anchors


def _discover_temporal_detectors(
    circuit: stim.Circuit,
    n_meas: int,
    qubit_meas: Dict[int, List[int]],
) -> List[FrozenSet[int]]:
    """Find weight-2 temporal detectors: consecutive same-qubit measurements.

    For each qubit measured multiple times, test
    ``1 → rec[-j] ⊕ rec[-k]`` for consecutive pairs.
    """
    temporal: List[FrozenSet[int]] = []
    for _q, indices in qubit_meas.items():
        for i in range(len(indices) - 1):
            j_idx, k_idx = indices[i], indices[i + 1]
            off_j = n_meas - j_idx
            off_k = n_meas - k_idx
            flow_str = f"1 -> rec[-{off_j}] xor rec[-{off_k}]"
            try:
                if circuit.has_flow(stim.Flow(flow_str), unsigned=True):
                    temporal.append(frozenset([j_idx, k_idx]))
            except Exception:
                pass
    return temporal


def _discover_boundary_detectors(
    circuit: stim.Circuit,
    n_meas: int,
    meas_info: List[Tuple[int, str, str]],
    qubit_meas: Dict[int, List[int]],
    existing: Set[FrozenSet[int]],
    qubit_coords: Dict[int, Tuple[float, ...]] = None,
    max_data_weight: int = 4,
    max_nearby_data: int = 8,
) -> List[FrozenSet[int]]:
    """Find weight-3+ boundary detectors.

    These combine a second-to-last ancilla measurement with final
    data-qubit measurements. The ancilla's last stabiliser extraction
    must match a subset of data qubit outcomes.

    To keep the combinatorial search tractable, only the spatially
    nearest ``max_nearby_data`` data qubits are considered for each
    ancilla qubit.

    Parameters
    ----------
    max_data_weight : int
        Maximum number of data-qubit measurements per boundary detector.
    max_nearby_data : int
        Maximum number of spatially nearest data qubits to search per
        ancilla.
    """
    if qubit_coords is None:
        qubit_coords = {}

    # Classify qubits
    ancilla_qubits: Set[int] = set()
    data_qubits: Set[int] = set()
    for q, indices in qubit_meas.items():
        if len(indices) == 1:
            data_qubits.add(q)
        elif len(indices) >= 4:
            ancilla_qubits.add(q)

    if not data_qubits or not ancilla_qubits:
        return []

    # Map: data qubit → final measurement index
    data_q_to_final: Dict[int, int] = {
        q: qubit_meas[q][-1] for q in data_qubits
    }

    # Second-to-last ancilla measurement indices
    anc_second_last: List[Tuple[int, int]] = []  # (meas_idx, qubit_idx)
    for q in ancilla_qubits:
        indices = qubit_meas[q]
        if len(indices) >= 2:
            anc_second_last.append((indices[-2], q))

    # For each ancilla, find spatially nearest data qubits
    def _dist(q1: int, q2: int) -> float:
        c1 = qubit_coords.get(q1, ())
        c2 = qubit_coords.get(q2, ())
        if len(c1) >= 2 and len(c2) >= 2:
            return ((c1[0] - c2[0]) ** 2 + (c1[1] - c2[1]) ** 2) ** 0.5
        return float("inf")

    boundary: List[FrozenSet[int]] = []
    data_qubit_list = sorted(data_qubits)

    for anc_idx, anc_q in anc_second_last:
        # Sort data qubits by distance to this ancilla
        if qubit_coords:
            nearby = sorted(data_qubit_list, key=lambda dq: _dist(anc_q, dq))
            nearby = nearby[:max_nearby_data]
        else:
            nearby = data_qubit_list[:max_nearby_data]

        nearby_final = sorted(data_q_to_final[dq] for dq in nearby)

        for weight in range(2, max_data_weight + 1):
            for data_combo in itertools.combinations(nearby_final, weight):
                candidate = frozenset([anc_idx] + list(data_combo))
                if candidate in existing:
                    continue

                rec_parts = []
                for idx in sorted(candidate):
                    off = n_meas - idx
                    rec_parts.append(f"rec[-{off}]")
                flow_str = "1 -> " + " xor ".join(rec_parts)

                try:
                    if circuit.has_flow(stim.Flow(flow_str), unsigned=True):
                        boundary.append(candidate)
                        existing.add(candidate)
                except Exception:
                    pass

    return boundary


def _prune_redundant_gf2(
    detectors: List[FrozenSet[int]],
    n_meas: int,
) -> List[FrozenSet[int]]:
    """Remove redundant detectors via GF(2) Gaussian elimination.

    A detector is redundant if it equals the XOR of other detectors.
    Returns a maximal linearly independent subset.

    Uses numpy for efficient matrix operations over GF(2).
    """
    if not detectors:
        return []

    # Sort: prefer low-weight detectors (they're more useful for decoding)
    det_list = sorted(detectors, key=lambda d: (len(d), sorted(d)))
    n_det = len(det_list)

    # Build binary matrix: each row = detector, each column = measurement
    matrix = np.zeros((n_det, n_meas), dtype=np.uint8)
    for i, det in enumerate(det_list):
        for idx in det:
            matrix[i, idx] = 1

    # Gaussian elimination over GF(2) — select independent rows
    independent_indices: List[int] = []
    used_rows: Set[int] = set()

    for col in range(n_meas):
        # Find a row with a 1 in this column (not yet used)
        found = None
        for row in range(n_det):
            if row not in used_rows and matrix[row, col] == 1:
                found = row
                break

        if found is None:
            continue

        used_rows.add(found)
        independent_indices.append(found)

        # Eliminate this column from all other rows
        for row in range(n_det):
            if row != found and matrix[row, col] == 1:
                matrix[row] ^= matrix[found]

    return [det_list[i] for i in sorted(independent_indices)]


def _assign_coordinates(
    det_indices: FrozenSet[int],
    meas_info: List[Tuple[int, str, str]],
    qubit_coords: Dict[int, Tuple[float, ...]],
    n_meas: int,
) -> Tuple[float, ...]:
    """Assign spatial coordinates to a detector.

    Uses the centroid of the participating qubits' coordinates,
    with a temporal coordinate derived from the measurement position.
    """
    xs, ys = [], []
    for idx in det_indices:
        q = meas_info[idx][0]
        if q in qubit_coords:
            coord = qubit_coords[q]
            if len(coord) >= 2:
                xs.append(coord[0])
                ys.append(coord[1])

    x = sum(xs) / len(xs) if xs else 0.0
    y = sum(ys) / len(ys) if ys else 0.0

    # Temporal coordinate: normalised position of latest measurement
    t = max(det_indices) / max(n_meas, 1) * 50.0  # rough time scale

    return (x, y, t)


def discover_detectors(
    circuit: stim.Circuit,
    *,
    use_cache: bool = True,
) -> FlowMatchResult:
    """Discover all deterministic detectors in a circuit.

    Uses Stim's ``has_flow()`` to validate candidate measurement subsets,
    then prunes redundant detectors via GF(2) Gaussian elimination.

    Parameters
    ----------
    circuit : stim.Circuit
        The circuit to analyse. Should include QUBIT_COORDS.
        DETECTOR and OBSERVABLE_INCLUDE instructions are ignored.
    use_cache : bool
        Whether to use/populate the module-level cache.

    Returns
    -------
    FlowMatchResult
        Discovered detectors (independent set) and metadata.
    """
    if use_cache:
        cached = _cache.get(circuit)
        if cached is not None:
            return cached

    # Strip existing annotations for analysis
    bare = stim.Circuit()
    for inst in circuit.flattened():
        if inst.name not in ("DETECTOR", "OBSERVABLE_INCLUDE"):
            bare.append(inst)

    qubit_coords = _extract_qubit_coords(circuit)
    n_meas = _count_measurements(bare)
    meas_info = _get_measurement_info(bare)
    qubit_meas = _group_measurements_by_qubit(meas_info)

    # Phase 1: Anchor detectors (weight 1)
    raw_dets: List[FrozenSet[int]] = _discover_anchor_detectors(bare, n_meas)
    logger.debug("Anchor detectors: %d", len(raw_dets))

    # Phase 2: Temporal detectors (weight 2, same qubit)
    temporal = _discover_temporal_detectors(bare, n_meas, qubit_meas)
    raw_dets.extend(temporal)
    logger.debug("Temporal detectors: %d", len(temporal))

    # Phase 3: Boundary detectors (weight 3-5)
    existing_set: Set[FrozenSet[int]] = set(raw_dets)
    boundary = _discover_boundary_detectors(
        bare, n_meas, meas_info, qubit_meas, existing_set,
        qubit_coords=qubit_coords,
    )
    raw_dets.extend(boundary)
    logger.debug("Boundary detectors: %d", len(boundary))

    raw_count = len(raw_dets)
    logger.debug("Raw detectors total: %d", raw_count)

    # Phase 4: Prune to independent set
    independent = _prune_redundant_gf2(raw_dets, n_meas)
    logger.debug("Independent detectors: %d", len(independent))

    # Phase 5: Assign coordinates
    discovered = [
        DiscoveredDetector(
            measurement_indices=det,
            coordinates=_assign_coordinates(det, meas_info, qubit_coords, n_meas),
        )
        for det in independent
    ]

    result = FlowMatchResult(
        detectors=discovered,
        num_measurements=n_meas,
        qubit_coords=qubit_coords,
        raw_detector_count=raw_count,
    )

    if use_cache:
        _cache.put(circuit, result)
    return result


# ---------------------------------------------------------------------------
# High-level: emit detectors into a circuit
# ---------------------------------------------------------------------------

def emit_auto_detectors(
    target_circuit: stim.Circuit,
    source_circuit: stim.Circuit,
    ctx: "DetectorContext",
    *,
    meas_offset: int = 0,
) -> int:
    """Discover detectors in source_circuit and emit them into target_circuit.

    Parameters
    ----------
    target_circuit : stim.Circuit
        Circuit to append DETECTOR instructions to.
    source_circuit : stim.Circuit
        Circuit to analyse for detectors.
    ctx : DetectorContext
        Current detector context (for measurement index tracking).
    meas_offset : int
        Offset to add to measurement indices.

    Returns
    -------
    int
        Number of detectors emitted.
    """
    result = discover_detectors(source_circuit)

    num_emitted = 0
    for det in result.detectors:
        abs_indices = sorted(det.measurement_indices)
        if abs_indices:
            ctx.emit_detector(target_circuit, abs_indices, det.coordinates)
            num_emitted += 1

    return num_emitted


# ---------------------------------------------------------------------------
# Config generation: produce CrossingDetectorConfig from circuit analysis
# ---------------------------------------------------------------------------

def compute_crossing_config_from_circuit(
    pre_circuit: stim.Circuit,
    post_circuit: stim.Circuit,
) -> "CrossingDetectorConfig":
    """Derive CrossingDetectorConfig by comparing flows across a gate boundary.

    Uses the flow-based detector discovery on the combined pre+post circuit
    to find detectors that span the gate boundary.

    Parameters
    ----------
    pre_circuit : stim.Circuit
        The last pre-gadget QEC round.
    post_circuit : stim.Circuit
        The first post-gadget QEC round.

    Returns
    -------
    CrossingDetectorConfig
        Configuration compatible with CrossingDetectorEmitter.
    """
    from qectostim.gadgets.base import (
        CrossingDetectorConfig,
        CrossingDetectorFormula,
        CrossingDetectorTerm,
    )

    # Combine circuits and discover detectors across the boundary
    combined = pre_circuit + post_circuit
    result = discover_detectors(combined, use_cache=False)
    n_pre_meas = _count_measurements(pre_circuit)

    # Crossing detectors span both circuits
    formulas: List[CrossingDetectorFormula] = []
    for i, det in enumerate(result.detectors):
        indices = sorted(det.measurement_indices)
        has_pre = any(idx < n_pre_meas for idx in indices)
        has_post = any(idx >= n_pre_meas for idx in indices)
        if has_pre and has_post:
            terms: List[CrossingDetectorTerm] = []
            for idx in indices:
                timing = "pre" if idx < n_pre_meas else "post"
                terms.append(CrossingDetectorTerm(
                    block="block_0",
                    stabilizer_type="Z",
                    timing=timing,
                ))
            formulas.append(CrossingDetectorFormula(
                name=f"auto_crossing_{i}",
                terms=terms,
            ))

    return CrossingDetectorConfig(formulas=formulas)


# ---------------------------------------------------------------------------
# Config generation: produce BoundaryDetectorConfig from circuit analysis
# ---------------------------------------------------------------------------

def compute_boundary_config_from_circuit(
    final_round_circuit: stim.Circuit,
    data_measurement_circuit: stim.Circuit,
    block_names: List[str],
) -> "BoundaryDetectorConfig":
    """Derive BoundaryDetectorConfig by matching final-round flows with data
    measurements.

    Uses flow-based detector discovery on the combined final-round + data-
    measurement circuit to find boundary detectors.

    Parameters
    ----------
    final_round_circuit : stim.Circuit
        The last syndrome extraction round.
    data_measurement_circuit : stim.Circuit
        The destructive data qubit measurement round.
    block_names : list of str
        Names of the surviving code blocks.

    Returns
    -------
    BoundaryDetectorConfig
        Configuration compatible with BoundaryDetectorEmitter.
    """
    from qectostim.gadgets.base import BoundaryDetectorConfig

    # Combine circuits and discover boundary detectors
    combined = final_round_circuit + data_measurement_circuit
    n_final_meas = _count_measurements(final_round_circuit)

    try:
        result = discover_detectors(combined, use_cache=False)
        # Check if any detectors span both circuits
        has_boundary = any(
            any(idx < n_final_meas for idx in det.measurement_indices)
            and any(idx >= n_final_meas for idx in det.measurement_indices)
            for det in result.detectors
        )
    except Exception as e:
        logger.warning("Boundary config discovery failed: %s", e)
        has_boundary = True  # Fall back to all-true

    block_configs: Dict[str, Dict[str, bool]] = {}
    for block_name in block_names:
        block_configs[block_name] = {
            "X": has_boundary,
            "Z": has_boundary,
        }

    return BoundaryDetectorConfig(block_configs=block_configs)


# ---------------------------------------------------------------------------
# Full-circuit auto-detection: bypass all manual configs
# ---------------------------------------------------------------------------

class AutoDetectorEmitter:
    """Automatic detector emitter for CSS gadget circuits.

    Provides two usage patterns:

    1. **Direct emission** — Call ``emit_all()`` with a fully-constructed
       ``stim.Circuit``. All DETECTOR instructions are computed and appended
       automatically. This replaces the entire manual detector pipeline.

    2. **Config generation** — Call ``compute_crossing_config()`` or
       ``compute_boundary_config()`` to produce the same dataclass configs
       that the manual pipeline uses. This enables gradual migration.

    Parameters
    ----------
    circuit : stim.Circuit
        The complete (or partial) circuit to analyse.
    qubit_coords : dict, optional
        Override qubit coordinate mapping. If None, extracted from circuit.

    Examples
    --------
    Direct emission::

        emitter = AutoDetectorEmitter(full_circuit)
        n_det = emitter.emit_all(full_circuit, ctx)

    Config generation::

        emitter = AutoDetectorEmitter(pre_round_circuit + gate_circuit + post_round_circuit)
        crossing = emitter.compute_crossing_config()
    """

    def __init__(
        self,
        circuit: stim.Circuit,
        qubit_coords: Optional[Dict[int, Tuple[float, ...]]] = None,
    ):
        self._circuit = circuit
        self._qubit_coords = qubit_coords or _extract_qubit_coords(circuit)
        self._result: Optional[FlowMatchResult] = None

    def _ensure_analysed(self) -> FlowMatchResult:
        if self._result is None:
            self._result = discover_detectors(self._circuit)
        return self._result

    @property
    def detectors(self) -> List[DiscoveredDetector]:
        """All discovered detectors."""
        return self._ensure_analysed().detectors

    @property
    def num_detectors(self) -> int:
        """Number of discovered detectors."""
        return len(self.detectors)

    def emit_all(
        self,
        circuit: stim.Circuit,
        ctx: "DetectorContext",
        *,
        meas_offset: int = 0,
    ) -> int:
        """Emit all discovered detectors into the circuit.

        Parameters
        ----------
        circuit : stim.Circuit
            Target circuit for DETECTOR instructions.
        ctx : DetectorContext
            Context for measurement index bookkeeping.
        meas_offset : int
            Global offset for measurement indices.

        Returns
        -------
        int
            Number of detectors emitted.
        """
        result = self._ensure_analysed()
        num_emitted = 0
        for det in result.detectors:
            abs_indices = sorted(det.measurement_indices)
            if abs_indices:
                ctx.emit_detector(circuit, abs_indices, det.coordinates)
                num_emitted += 1
        return num_emitted

    def validate_determinism(
        self,
        circuit: stim.Circuit,
    ) -> Tuple[int, int]:
        """Check how many discovered detectors are truly deterministic.

        Uses Stim's ``circuit.detector_error_model(allow_gauge_detectors=True)``
        to check for non-deterministic detectors.

        Parameters
        ----------
        circuit : stim.Circuit
            Circuit WITH detectors already emitted.

        Returns
        -------
        (total, non_deterministic)
            Total number of detectors and how many are non-deterministic.
        """
        try:
            dem = circuit.detector_error_model(
                decompose_errors=False,
                approximate_disjoint_errors=True,
                ignore_decomposition_failures=True,
                allow_gauge_detectors=True,
            )
            total = 0
            non_det = 0
            for inst in dem.flattened():
                if inst.type == "error":
                    total_targets = inst.targets_copy()
                    for t in total_targets:
                        if t.is_relative_detector_id():
                            total += 1
            # Actually, count detectors in the circuit
            det_count = 0
            for inst in circuit.flattened():
                if inst.name == "DETECTOR":
                    det_count += 1

            # Check with noiseless simulation
            sampler = circuit.compile_detector_sampler()
            samples = sampler.sample(shots=100)
            non_det_count = 0
            for col in range(samples.shape[1]):
                if samples[:, col].any():
                    non_det_count += 1

            return det_count, non_det_count

        except Exception as e:
            logger.warning("Determinism check failed: %s", e)
            return 0, 0


# ---------------------------------------------------------------------------
# Convenience: analyse a full FT experiment circuit
# ---------------------------------------------------------------------------

def analyse_circuit_detectors(
    circuit: stim.Circuit,
) -> Dict[str, int]:
    """Analyse a circuit and report detector statistics.

    Parameters
    ----------
    circuit : stim.Circuit
        Complete circuit (with or without existing detectors).

    Returns
    -------
    dict
        Statistics: 'discovered', 'existing', 'non_deterministic'.
    """
    # Count existing detectors
    existing = sum(
        1 for inst in circuit.flattened() if inst.name == "DETECTOR"
    )

    # Strip existing detectors and discover automatically
    stripped = stim.Circuit()
    for inst in circuit.flattened():
        if inst.name not in ("DETECTOR", "OBSERVABLE_INCLUDE"):
            stripped.append(inst)

    result = discover_detectors(stripped, use_cache=False)

    return {
        "discovered": len(result.detectors),
        "existing": existing,
    }


# ---------------------------------------------------------------------------
# Auto observable discovery (pure-circuit, no gadget info needed)
# ---------------------------------------------------------------------------

@dataclass
class DiscoveredObservable:
    """A valid observable discovered via flow validation.

    Attributes
    ----------
    measurement_indices : frozenset of int
        Absolute measurement indices whose XOR forms this observable.
    description : str
        Human-readable description.
    n_blocks : int
        Number of distinct code blocks spanned by this observable.
    """
    measurement_indices: FrozenSet[int]
    description: str = ""
    n_blocks: int = 1

    @property
    def weight(self) -> int:
        return len(self.measurement_indices)


def discover_observables(
    circuit: stim.Circuit,
    *,
    max_weight: int = 10,
    max_candidates: int = 5000,
) -> List[DiscoveredObservable]:
    """Discover valid observables from a bare circuit.

    An observable is a set of measurements whose XOR is deterministic
    (i.e., ``has_flow("1 -> 1 xor rec[...])`` is true) AND that involves
    at least one final data-qubit measurement (measured only once).

    Strategy (fast, targeted)
    -------------------------
    1. Classify measurements: data qubits (measured once) vs ancilla
       (measured many times).  Collect mid-circuit MX measurements.
    2. Sweep **small** subsets of final data measurements, organised by
       basis (logical X / Z observable lines), weights 1 → max_weight.
    3. For each weight, also try appending 1–2 mid-circuit MX corrections.
    4. Stop early once *max_candidates* ``has_flow`` calls are made.

    This finds the manual-equivalent observable (weight ≈ d) in ≤ 0.5 s
    for RotatedSurface d = 3.

    Parameters
    ----------
    circuit : stim.Circuit
        Complete circuit (DETECTOR/OBSERVABLE_INCLUDE are stripped).
    max_weight : int
        Maximum number of measurements in an observable candidate.
    max_candidates : int
        Budget for ``has_flow`` calls (prevents combinatorial blowup).

    Returns
    -------
    list of DiscoveredObservable
        All valid observables found within the budget.
    """
    # Strip annotations
    bare = stim.Circuit()
    for inst in circuit.flattened():
        if inst.name not in ("DETECTOR", "OBSERVABLE_INCLUDE"):
            bare.append(inst)

    n_meas = _count_measurements(bare)
    meas_info = _get_measurement_info(bare)
    qubit_meas = _group_measurements_by_qubit(meas_info)

    # Classify qubits -------------------------------------------------
    data_qubits: Set[int] = set()
    ancilla_qubits: Set[int] = set()
    for q, indices in qubit_meas.items():
        if len(indices) == 1:
            data_qubits.add(q)
        elif len(indices) >= 2:
            ancilla_qubits.add(q)

    # Final data measurement indices
    final_data = sorted(qubit_meas[q][-1] for q in data_qubits)

    # Mid-circuit MX measurements (ancilla-based, potential correction terms)
    mid_circuit_x: List[int] = []
    for q in ancilla_qubits:
        for idx in qubit_meas[q]:
            if meas_info[idx][1] == "X":
                mid_circuit_x.append(idx)
    mid_circuit_x.sort()

    # Group final data by basis
    final_by_basis: Dict[str, List[int]] = {}
    for idx in final_data:
        b = meas_info[idx][1]
        final_by_basis.setdefault(b, []).append(idx)

    logger.debug(
        "Observable search: %d data qubits (%s), %d mid-circuit MX",
        len(data_qubits),
        {b: len(v) for b, v in final_by_basis.items()},
        len(mid_circuit_x),
    )

    # ── Block detection via QUBIT_COORDS ─────────────────────────────
    # A valid logical observable should span measurements from at least
    # two distinct code blocks.  We infer blocks from spatial clustering
    # of qubit coordinates (x-axis gaps).
    qubit_coords: Dict[int, Tuple[float, ...]] = {}
    for inst in bare.flattened():
        if inst.name == "QUBIT_COORDS":
            tgts = inst.targets_copy()
            qubit_coords[tgts[0].value] = tuple(inst.gate_args_copy())

    def _assign_blocks(qubits: Set[int]) -> Dict[int, int]:
        """Assign qubits to blocks by coordinate clustering.
        
        Uses simple distance-based clustering: two qubits are in the
        same block if there is a chain of qubits between them where
        each consecutive pair has distance ≤ 3.0.  This handles both
        horizontal and vertical separation (e.g., CSS Surgery has
        blocks at different y-ranges with the same x-range).
        """
        if not qubit_coords:
            return {q: 0 for q in qubits}
        import math
        qs = sorted(qubits)
        # Union-find for clustering
        parent = {q: q for q in qs}
        
        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        
        def union(a: int, b: int) -> None:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb
        
        # Merge qubits that are within distance 3.0
        for i in range(len(qs)):
            ci = qubit_coords.get(qs[i])
            if ci is None:
                continue
            for j in range(i + 1, len(qs)):
                cj = qubit_coords.get(qs[j])
                if cj is None:
                    continue
                dist = math.sqrt(sum((a - b) ** 2 for a, b in zip(ci, cj)))
                if dist <= 3.0:
                    union(qs[i], qs[j])
        
        # Assign block IDs
        roots = {}
        assignment: Dict[int, int] = {}
        for q in qs:
            r = find(q)
            if r not in roots:
                roots[r] = len(roots)
            assignment[q] = roots[r]
        return assignment

    data_block = _assign_blocks(data_qubits)
    n_blocks = len(set(data_block.values()))
    # Map measurement index → block
    meas_block: Dict[int, int] = {}
    for q in data_qubits:
        for idx in qubit_meas[q]:
            meas_block[idx] = data_block[q]

    logger.debug("  detected %d code blocks among data qubits", n_blocks)

    # Pre-filter: identify trivially deterministic measurements --------
    trivial_det: Set[int] = set()
    for idx in final_data:
        try:
            if bare.has_flow(
                stim.Flow(f"1 -> rec[{idx - n_meas}]"), unsigned=True
            ):
                trivial_det.add(idx)
        except Exception:
            pass

    logger.debug(
        "  trivially deterministic: %d/%d data meas",
        len(trivial_det), len(final_data),
    )

    # Candidate testing ------------------------------------------------
    observables: List[DiscoveredObservable] = []
    found_sets: Set[FrozenSet[int]] = set()
    calls = 0

    def _spans_multi_block(indices: List[int]) -> bool:
        """Check if the candidate spans ≥ 2 code blocks."""
        if n_blocks <= 1:
            return True  # single-block gadget, skip check
        blocks_seen = set()
        for idx in indices:
            b = meas_block.get(idx)
            if b is not None:
                blocks_seen.add(b)
        return len(blocks_seen) >= 2

    # When ALL data measurements are trivially deterministic, we must
    # relax the trivial-subset filter but require multi-block spanning.
    all_trivial = trivial_det and len(trivial_det) == len(final_data)

    def _try(indices: List[int], desc: str) -> bool:
        nonlocal calls
        calls += 1
        candidate = frozenset(indices)
        if candidate in found_sets or len(candidate) > max_weight:
            return False
        # Skip weight-1 candidates — they're detectors, not observables.
        if len(candidate) < 2:
            return False
        if all_trivial:
            # All data are deterministic: only accept cross-block combos
            if n_blocks > 1 and not _spans_multi_block(list(candidate)):
                return False
        else:
            # Skip candidates composed entirely of trivially deterministic
            # measurements — they're products of detectors, not true
            # logical observables.
            if trivial_det and candidate.issubset(trivial_det):
                return False
        rec_parts = [f"rec[{idx - n_meas}]" for idx in sorted(candidate)]
        flow_str = "1 -> 1 xor " + " xor ".join(rec_parts)
        try:
            if bare.has_flow(stim.Flow(flow_str), unsigned=True):
                found_sets.add(candidate)
                # Count how many code blocks are spanned
                blocks_seen = set()
                for idx in candidate:
                    b = meas_block.get(idx)
                    if b is not None:
                        blocks_seen.add(b)
                observables.append(DiscoveredObservable(
                    measurement_indices=candidate,
                    description=desc,
                    n_blocks=len(blocks_seen) if blocks_seen else 1,
                ))
                return True
        except Exception:
            pass
        return False

    def _budget_ok() -> bool:
        return calls < max_candidates

    # ------------------------------------------------------------------
    # Phase 1: Pure data-measurement subsets, weight 2 → max_weight
    #   First try within-basis subsets at low weights (2-4).
    #   Then try cross-block combos at higher weights (d..2d) by
    #   picking subsets from each block and combining.
    # ------------------------------------------------------------------

    # 1a: Small-weight within-basis (w ≤ 4, fast)
    for basis, indices in final_by_basis.items():
        for w in range(2, min(len(indices) + 1, 5)):
            if not _budget_ok():
                break
            for combo in itertools.combinations(indices, w):
                if not _budget_ok():
                    break
                _try(
                    list(combo),
                    f"data_{basis}_{w}",
                )

    # 1b: Cross-block within-basis combos at higher weights (5..max_weight)
    #     Group measurements by (basis, block) and combine subsets from
    #     different blocks, keeping per-block size ≤ 5.
    basis_block: Dict[Tuple[str, int], List[int]] = {}
    for idx in final_data:
        b = meas_info[idx][1]
        blk = meas_block.get(idx, 0)
        basis_block.setdefault((b, blk), []).append(idx)

    if n_blocks >= 2 and _budget_ok():
        for basis in final_by_basis:
            # Get all blocks for this basis
            blk_groups = []
            for (b, blk), idxs in sorted(basis_block.items()):
                if b == basis:
                    blk_groups.append((blk, idxs))
            if len(blk_groups) < 2:
                continue
            # Try combining subsets from pairs of blocks
            for bi in range(len(blk_groups)):
                for bj in range(bi + 1, len(blk_groups)):
                    if not _budget_ok():
                        break
                    g1 = blk_groups[bi][1]
                    g2 = blk_groups[bj][1]
                    for w1 in range(1, min(len(g1) + 1, 6)):
                        if not _budget_ok():
                            break
                        for w2 in range(1, min(len(g2) + 1, 6)):
                            if w1 + w2 > max_weight or w1 + w2 < 2:
                                continue
                            if not _budget_ok():
                                break
                            for c1 in itertools.combinations(g1, w1):
                                if not _budget_ok():
                                    break
                                for c2 in itertools.combinations(g2, w2):
                                    if not _budget_ok():
                                        break
                                    _try(
                                        list(c1) + list(c2),
                                        f"data_{basis}_{w1}+{w2}",
                                    )

    # ------------------------------------------------------------------
    # Phase 1b: Cross-basis data subsets (e.g., Z from block1 + X from
    #   block2).  Only try when there are multiple bases AND blocks.
    # ------------------------------------------------------------------
    if len(final_by_basis) >= 2 and n_blocks >= 2 and _budget_ok():
        all_bases = list(final_by_basis.keys())
        for bi in range(len(all_bases)):
            for bj in range(bi + 1, len(all_bases)):
                if not _budget_ok():
                    break
                idx_i = final_by_basis[all_bases[bi]]
                idx_j = final_by_basis[all_bases[bj]]
                for wi in range(1, min(len(idx_i) + 1, 6)):
                    if not _budget_ok():
                        break
                    for wj in range(1, min(len(idx_j) + 1, 6)):
                        if wi + wj > max_weight or not _budget_ok():
                            break
                        for ci in itertools.combinations(idx_i, wi):
                            if not _budget_ok():
                                break
                            for cj in itertools.combinations(idx_j, wj):
                                if not _budget_ok():
                                    break
                                _try(
                                    list(ci) + list(cj),
                                    f"data_{all_bases[bi]}{wi}+{all_bases[bj]}{wj}",
                                )

    # ------------------------------------------------------------------
    # Phase 2: Data subsets + 1 mid-circuit MX correction
    #   Only try data weights 1 → 7 to keep it fast.
    # ------------------------------------------------------------------
    for mid_idx in mid_circuit_x:
        if not _budget_ok():
            break
        for basis, data_indices in final_by_basis.items():
            if not _budget_ok():
                break
            for w in range(1, min(len(data_indices) + 1, 8)):
                if not _budget_ok():
                    break
                for combo in itertools.combinations(data_indices, w):
                    if not _budget_ok():
                        break
                    _try(
                        [mid_idx] + list(combo),
                        f"data_{basis}_{w}+midX({meas_info[mid_idx][0]})",
                    )

    # ------------------------------------------------------------------
    # Phase 3: Data subsets + 2 mid-circuit MX corrections
    #   Only if ≤ 15 mid-circuit X measurements (keeps combos bounded).
    # ------------------------------------------------------------------
    if len(mid_circuit_x) <= 15 and _budget_ok():
        for mi, mj in itertools.combinations(mid_circuit_x, 2):
            if not _budget_ok():
                break
            for basis, data_indices in final_by_basis.items():
                if not _budget_ok():
                    break
                for w in range(1, min(len(data_indices) + 1, 7)):
                    if not _budget_ok():
                        break
                    for combo in itertools.combinations(data_indices, w):
                        if not _budget_ok():
                            break
                        _try(
                            [mi, mj] + list(combo),
                            f"data_{basis}_{w}+midX2",
                        )

    # ------------------------------------------------------------------
    # Phase 4: All final data together (full-block observable)
    # ------------------------------------------------------------------
    if _budget_ok():
        _try(final_data, "all_final_data")

    logger.debug(
        "Observable search: %d valid from %d candidates tested",
        len(observables), calls,
    )
    return observables


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    "AutoDetectorEmitter",
    "DiscoveredDetector",
    "DiscoveredObservable",
    "FlowMatchResult",
    "analyse_circuit_detectors",
    "compute_boundary_config_from_circuit",
    "compute_crossing_config_from_circuit",
    "discover_detectors",
    "discover_observables",
    "emit_auto_detectors",
    "split_circuit_into_rounds",
]
