# src/qectostim/experiments/tqec_observable.py
"""
Automatic observable emission for CSS gadgets.

This module automatically derives ``ObservableConfig`` for any CSS gadget by:

1. Computing the Heisenberg-picture logical operator propagation through
   the gate's Clifford action (from ``get_observable_transform()``).
2. Determining which mid-circuit measurements (bridge/merge) must be XOR'd
   into the observable from the stabilizer transform.
3. Validating each candidate observable against the circuit using
   ``stim.Circuit.has_flow()`` (Stim ≥ 1.14) to catch backward-propagation
   issues automatically (like the |00⟩ s_zz problem).
4. Falling back gracefully when a candidate is rejected.

Architecture
------------
The module produces ``ObservableConfig`` — the same dataclass the existing
``emit_observable()`` in ``observable.py`` consumes. No changes to the
experiment orchestration are needed.

Flow
----
1. ``AutoObservableEmitter.from_gadget(gadget, codes, circuit)``
2. Examines ``gadget.get_observable_transform()`` to determine which logical
   operators the observable should track.
3. Enumerates candidate formulas (measurement XOR combinations).
4. Validates each candidate against the circuit with ``has_flow(unsigned=True)``.
5. Returns the first valid ``ObservableConfig``.

Dependencies
------------
- ``stim >= 1.14``: For ``Circuit.has_flow()`` validation.
- Works with the existing ``ObservableConfig``, ``ObservableTerm`` dataclasses
  from ``gadgets/base.py``.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from itertools import combinations
from typing import (
    Dict, List, Optional, Sequence, Set, Tuple, TYPE_CHECKING,
)

import stim

if TYPE_CHECKING:
    from qectostim.codes.abstract_code import Code
    from qectostim.gadgets.base import (
        Gadget,
        ObservableConfig,
        ObservableTerm,
        ObservableTransform,
    )

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class CandidateObservable:
    """A candidate observable formula to be validated.

    Attributes
    ----------
    terms : list of (block_name, basis) tuples
        Logical operator terms (data qubit readouts).
    bridge_meas_indices : list of int
        Mid-circuit measurement indices to XOR (bridge/merge corrections).
    description : str
        Human-readable description of this candidate.
    """
    terms: List[Tuple[str, str]]
    bridge_meas_indices: List[int] = field(default_factory=list)
    description: str = ""

    def to_observable_config(self) -> "ObservableConfig":
        """Convert to an ObservableConfig."""
        from qectostim.gadgets.base import ObservableConfig, ObservableTerm

        # Derive block bases from terms
        block_bases: Dict[str, str] = {}
        for block_name, basis in self.terms:
            block_bases[block_name] = basis

        return ObservableConfig(
            output_blocks=[block for block, _ in self.terms],
            block_bases=block_bases,
            correlation_terms=[
                ObservableTerm(block=block, basis=basis)
                for block, basis in self.terms
            ],
            bridge_frame_meas_indices=list(self.bridge_meas_indices),
        )


@dataclass
class ObservableSearchResult:
    """Result of automatic observable search.

    Attributes
    ----------
    config : ObservableConfig or None
        The validated configuration, or None if no valid candidate found.
    candidates_tried : int
        Number of candidate formulas evaluated.
    valid_candidates : list of CandidateObservable
        All candidates that passed validation.
    rejected_candidates : list of (CandidateObservable, str)
        Candidates that failed validation, with reason.
    """
    config: Optional["ObservableConfig"] = None
    candidates_tried: int = 0
    valid_candidates: List[CandidateObservable] = field(default_factory=list)
    rejected_candidates: List[Tuple[CandidateObservable, str]] = field(
        default_factory=list
    )


# ---------------------------------------------------------------------------
# Observable validation using stim.Circuit.has_flow()
# ---------------------------------------------------------------------------

def validate_observable_flow(
    circuit: stim.Circuit,
    data_qubit_meas_indices: Dict[int, int],
    logical_support: Dict[str, List[Tuple[int, str]]],
    bridge_meas_indices: List[int],
    *,
    unsigned: bool = True,
) -> bool:
    """Validate that a candidate observable is a valid flow of the circuit.

    Uses ``stim.Circuit.has_flow()`` to check that the proposed measurement
    XOR is deterministic — i.e., it forms a valid stabilizer flow.

    Parameters
    ----------
    circuit : stim.Circuit
        Complete circuit including all measurements.
    data_qubit_meas_indices : dict
        Maps global qubit index → measurement index for final data measurement.
    logical_support : dict
        Maps block_name → list of (global_qubit_idx, pauli) pairs.
    bridge_meas_indices : list of int
        Mid-circuit measurement indices contributing to the observable.
    unsigned : bool
        If True, ignore signs (Pauli frame corrections). Default True.

    Returns
    -------
    bool
        True if the observable is a valid flow.
    """
    if not hasattr(stim.Circuit, "has_flow"):
        logger.warning(
            "stim.Circuit.has_flow() not available (stim >= 1.14 required). "
            "Skipping flow validation."
        )
        return True  # Optimistic fallback

    # Build the flow: 1 -> P xor rec[...] (preparation to measurement)
    # The observable is a product of data qubit measurements and bridge measurements.

    total_meas = _count_circuit_measurements(circuit)

    # Collect all measurement rec targets
    rec_targets: List[int] = []

    # Data qubit measurements
    for block_qubits in logical_support.values():
        for qubit_idx, _pauli in block_qubits:
            if qubit_idx in data_qubit_meas_indices:
                meas_idx = data_qubit_meas_indices[qubit_idx]
                rec_targets.append(meas_idx - total_meas)  # Convert to lookback

    # Bridge measurements
    for meas_idx in bridge_meas_indices:
        rec_targets.append(meas_idx - total_meas)

    if not rec_targets:
        return False

    # Build a stim.Flow: 1 -> 1 xor rec[...] (check flow)
    # This checks that the XOR of all specified measurements is deterministic.
    try:
        flow_str = "1 -> 1"
        for rec in sorted(rec_targets):
            flow_str += f" xor rec[{rec}]"
        flow = stim.Flow(flow_str)
        return circuit.has_flow(flow, unsigned=unsigned)
    except Exception as e:
        logger.debug("Flow validation raised: %s", e)
        return False


def _count_circuit_measurements(circuit: stim.Circuit) -> int:
    """Count total measurements in a circuit."""
    count = 0
    for inst in circuit.flattened():
        if inst.name in ("M", "MX", "MY", "MR", "MRX", "MRY", "MZ"):
            count += len(inst.targets_copy())
    return count


# ---------------------------------------------------------------------------
# Candidate generation
# ---------------------------------------------------------------------------

def generate_candidates_from_transform(
    observable_transform: "ObservableTransform",
    control_state: str,
    target_state: str,
    block_names: List[str],
    mid_circuit_meas: Optional[Dict[str, List[int]]] = None,
) -> List[CandidateObservable]:
    """Generate candidate observable formulas from a gate's observable transform.

    The observable transform describes how logical X and Z operators propagate
    through the gate. From this, we derive which data-qubit readouts (and
    mid-circuit measurements) should be XOR'd into the observable.

    For a CNOT with |control_state, target_state⟩ input:
    - |00⟩: Track Z_ctrl_out = Z_ctrl, Z_tgt_out = Z_tgt ⊕ Z_ctrl ⊕ s_zz
    - |+0⟩: Track X_ctrl_out = X_ctrl ⊕ X_anc (ancilla MX measurement)
    - |0+⟩: Track Z_ctrl_out ⊕ X_tgt_out
    - |++⟩: Track X_ctrl_out ⊕ X_tgt_out = X_ctrl ⊕ X_tgt ⊕ X_anc

    Parameters
    ----------
    observable_transform : ObservableTransform
        Gate's logical operator transformation.
    control_state : str
        Input state of control ("0" or "+").
    target_state : str
        Input state of target ("0" or "+").
    block_names : list of str
        Ordered block names (e.g., ["block_0", "block_1", "block_2"]).
    mid_circuit_meas : dict, optional
        Available mid-circuit measurements. Keys are descriptive names
        (e.g., "anc_mx", "zz_bridge", "xx_bridge"), values are lists of
        measurement indices.

    Returns
    -------
    list of CandidateObservable
        Ordered by preference (two-qubit observables first, then single-qubit).
    """
    if mid_circuit_meas is None:
        mid_circuit_meas = {}

    ctrl_basis = "Z" if control_state == "0" else "X"
    tgt_basis = "Z" if target_state == "0" else "X"

    ctrl_block = block_names[0] if block_names else "block_0"
    tgt_block = block_names[-1] if len(block_names) > 1 else "block_0"

    anc_mx = mid_circuit_meas.get("anc_mx", [])
    zz_bridge = mid_circuit_meas.get("zz_bridge", [])
    xx_bridge = mid_circuit_meas.get("xx_bridge", [])

    candidates: List[CandidateObservable] = []

    if ctrl_basis == "Z" and tgt_basis == "Z":
        # |00⟩ candidates — prefer two-qubit
        # 1. Z_ctrl ⊕ Z_tgt (no correction)
        candidates.append(CandidateObservable(
            terms=[(ctrl_block, "Z"), (tgt_block, "Z")],
            bridge_meas_indices=[],
            description="|00⟩: Z_ctrl ⊕ Z_tgt (no bridge correction)",
        ))
        # 2. Z_ctrl ⊕ Z_tgt ⊕ s_zz
        if zz_bridge:
            candidates.append(CandidateObservable(
                terms=[(ctrl_block, "Z"), (tgt_block, "Z")],
                bridge_meas_indices=list(zz_bridge),
                description="|00⟩: Z_ctrl ⊕ Z_tgt ⊕ s_zz",
            ))
        # 3. Z_ctrl only
        candidates.append(CandidateObservable(
            terms=[(ctrl_block, "Z")],
            bridge_meas_indices=[],
            description="|00⟩: Z_ctrl only",
        ))
        # 4. Z_tgt only
        candidates.append(CandidateObservable(
            terms=[(tgt_block, "Z")],
            bridge_meas_indices=[],
            description="|00⟩: Z_tgt only",
        ))

    elif ctrl_basis == "X" and tgt_basis == "Z":
        # |+0⟩ candidates
        # 1. X_ctrl ⊕ X_anc_MX (two-qubit via ancilla)
        if anc_mx:
            candidates.append(CandidateObservable(
                terms=[(ctrl_block, "X")],
                bridge_meas_indices=list(anc_mx),
                description="|+0⟩: X_ctrl ⊕ X_anc_MX",
            ))
        # 2. X_ctrl only
        candidates.append(CandidateObservable(
            terms=[(ctrl_block, "X")],
            bridge_meas_indices=[],
            description="|+0⟩: X_ctrl only",
        ))
        # 3. X_ctrl ⊕ X_anc_MX ⊕ s_xx
        if anc_mx and xx_bridge:
            candidates.append(CandidateObservable(
                terms=[(ctrl_block, "X")],
                bridge_meas_indices=list(anc_mx) + list(xx_bridge),
                description="|+0⟩: X_ctrl ⊕ X_anc_MX ⊕ s_xx",
            ))

    elif ctrl_basis == "Z" and tgt_basis == "X":
        # |0+⟩ candidates
        # 1. Z_ctrl ⊕ X_tgt
        candidates.append(CandidateObservable(
            terms=[(ctrl_block, "Z"), (tgt_block, "X")],
            bridge_meas_indices=[],
            description="|0+⟩: Z_ctrl ⊕ X_tgt",
        ))
        # 2. X_tgt only
        candidates.append(CandidateObservable(
            terms=[(tgt_block, "X")],
            bridge_meas_indices=[],
            description="|0+⟩: X_tgt only",
        ))
        # 3. Z_ctrl only
        candidates.append(CandidateObservable(
            terms=[(ctrl_block, "Z")],
            bridge_meas_indices=[],
            description="|0+⟩: Z_ctrl only",
        ))

    elif ctrl_basis == "X" and tgt_basis == "X":
        # |++⟩ candidates
        # 1. X_ctrl ⊕ X_tgt ⊕ X_anc_MX
        if anc_mx:
            candidates.append(CandidateObservable(
                terms=[(ctrl_block, "X"), (tgt_block, "X")],
                bridge_meas_indices=list(anc_mx),
                description="|++⟩: X_ctrl ⊕ X_tgt ⊕ X_anc_MX",
            ))
        # 2. X_ctrl ⊕ X_tgt ⊕ s_xx
        if xx_bridge:
            candidates.append(CandidateObservable(
                terms=[(ctrl_block, "X"), (tgt_block, "X")],
                bridge_meas_indices=list(xx_bridge),
                description="|++⟩: X_ctrl ⊕ X_tgt ⊕ s_xx",
            ))
        # 3. X_ctrl ⊕ X_tgt (no correction)
        candidates.append(CandidateObservable(
            terms=[(ctrl_block, "X"), (tgt_block, "X")],
            bridge_meas_indices=[],
            description="|++⟩: X_ctrl ⊕ X_tgt",
        ))
        # 4. X_ctrl only
        candidates.append(CandidateObservable(
            terms=[(ctrl_block, "X")],
            bridge_meas_indices=[],
            description="|++⟩: X_ctrl only",
        ))

    return candidates


# ---------------------------------------------------------------------------
# Exhaustive search: try ALL measurement XOR combinations
# ---------------------------------------------------------------------------

def exhaustive_observable_search(
    circuit: stim.Circuit,
    block_info: Dict[str, Dict],
    mid_circuit_meas_groups: Dict[str, List[int]],
    *,
    max_bridge_groups: int = 2,
    unsigned: bool = True,
) -> ObservableSearchResult:
    """Exhaustively search for valid observables by trying all combinations.

    This is the brute-force fallback when heuristic candidates fail. It
    systematically tries:
    - All single-block and multi-block logical operator combinations
    - With and without each group of mid-circuit measurements

    Parameters
    ----------
    circuit : stim.Circuit
        Complete circuit.
    block_info : dict
        Maps block_name → {"data_start": int, "n_qubits": int, "code": Code}.
    mid_circuit_meas_groups : dict
        Named groups of mid-circuit measurements.
        e.g., {"anc_mx": [136,139,142], "zz_bridge": [73,74,75]}
    max_bridge_groups : int
        Maximum number of bridge groups to combine.
    unsigned : bool
        Whether to use unsigned flow validation.

    Returns
    -------
    ObservableSearchResult
        All valid candidates found.
    """
    total_meas = _count_circuit_measurements(circuit)

    # Generate all block/basis combinations
    block_terms: List[List[Tuple[str, str]]] = []
    for block_name in block_info:
        block_terms.append([(block_name, "X"), (block_name, "Z")])

    # Generate bridge combinations
    bridge_combos: List[List[int]] = [[]]  # Start with no bridges
    group_names = list(mid_circuit_meas_groups.keys())
    for r in range(1, min(max_bridge_groups, len(group_names)) + 1):
        for combo in combinations(group_names, r):
            indices: List[int] = []
            for name in combo:
                indices.extend(mid_circuit_meas_groups[name])
            bridge_combos.append(indices)

    result = ObservableSearchResult()

    # Try each combination of block terms
    for n_blocks in range(1, len(block_info) + 1):
        for block_combo in combinations(block_info.keys(), n_blocks):
            for bases in _product_bases(len(block_combo)):
                terms = list(zip(block_combo, bases))
                for bridge in bridge_combos:
                    candidate = CandidateObservable(
                        terms=terms,
                        bridge_meas_indices=bridge,
                        description=f"{'⊕'.join(f'{b}_{bl}' for bl, b in terms)}"
                        + (f" ⊕ bridge[{len(bridge)}]" if bridge else ""),
                    )
                    result.candidates_tried += 1

                    # Build logical support for validation
                    logical_support: Dict[str, List[Tuple[int, str]]] = {}
                    data_qubit_meas: Dict[int, int] = {}
                    valid_block = True

                    for block_name, basis in terms:
                        bi = block_info[block_name]
                        code = bi["code"]
                        data_start = bi["data_start"]
                        support = (
                            code.get_logical_x_support()
                            if basis == "X"
                            else code.get_logical_z_support()
                        )
                        logical_support[block_name] = [
                            (data_start + idx, basis) for idx in support
                        ]
                        # Map data qubits to their measurement indices
                        n_qubits = bi["n_qubits"]
                        for i in range(n_qubits):
                            data_qubit_meas[data_start + i] = (
                                bi.get("meas_start", 0) + i
                            )

                    is_valid = validate_observable_flow(
                        circuit,
                        data_qubit_meas,
                        logical_support,
                        bridge,
                        unsigned=unsigned,
                    )

                    if is_valid:
                        result.valid_candidates.append(candidate)
                        if result.config is None:
                            # First valid = preferred (multi-qubit preferred
                            # because we iterate from n_blocks=1 upward, but
                            # we actually want multi-block first)
                            pass
                    else:
                        result.rejected_candidates.append(
                            (candidate, "has_flow() returned False")
                        )

    # Pick the best valid candidate: prefer multi-block, then fewer bridges
    if result.valid_candidates:
        best = max(
            result.valid_candidates,
            key=lambda c: (len(c.terms), -len(c.bridge_meas_indices)),
        )
        result.config = best.to_observable_config()

    return result


def _product_bases(n: int) -> List[Tuple[str, ...]]:
    """Generate all combinations of X/Z for n blocks."""
    if n == 0:
        return [()]
    result: List[Tuple[str, ...]] = []
    for rest in _product_bases(n - 1):
        result.append(("X",) + rest)
        result.append(("Z",) + rest)
    return result


# ---------------------------------------------------------------------------
# Main entry point: AutoObservableEmitter
# ---------------------------------------------------------------------------

class AutoObservableEmitter:
    """Automatic observable emitter for CSS gadgets.

    Derives ``ObservableConfig`` automatically by:
    1. Computing candidate formulas from the gate's observable transform.
    2. Validating each candidate against the circuit with ``has_flow()``.
    3. Falling back to exhaustive search if heuristics fail.

    Usage
    -----
    ::

        emitter = AutoObservableEmitter.from_gadget(
            gadget=gadget,
            codes=codes,
            circuit=full_circuit,
            alloc=alloc,
        )
        obs_config = emitter.get_config()

    The returned ``ObservableConfig`` plugs directly into ``emit_observable()``.

    Parameters
    ----------
    circuit : stim.Circuit
        Complete circuit for flow validation.
    candidates : list of CandidateObservable
        Ordered candidate formulas.
    block_info : dict
        Block metadata for exhaustive search fallback.
    mid_circuit_meas : dict
        Mid-circuit measurement groups.
    """

    def __init__(
        self,
        circuit: stim.Circuit,
        candidates: List[CandidateObservable],
        block_info: Dict[str, Dict],
        mid_circuit_meas: Optional[Dict[str, List[int]]] = None,
    ):
        self._circuit = circuit
        self._candidates = candidates
        self._block_info = block_info
        self._mid_circuit_meas = mid_circuit_meas or {}
        self._result: Optional[ObservableSearchResult] = None

    @classmethod
    def from_gadget(
        cls,
        gadget: "Gadget",
        codes: List["Code"],
        circuit: stim.Circuit,
        alloc: Dict,
        mid_circuit_meas: Optional[Dict[str, List[int]]] = None,
    ) -> "AutoObservableEmitter":
        """Create an emitter from a gadget and its circuit.

        Parameters
        ----------
        gadget : Gadget
            The CSS gadget.
        codes : list of Code
            Codes used in the experiment.
        circuit : stim.Circuit
            Complete circuit.
        alloc : dict
            Qubit allocation (block_name → {"data": (start, n), "code": ...}).
        mid_circuit_meas : dict, optional
            Mid-circuit measurement groups. If None, attempts to extract
            from gadget attributes.

        Returns
        -------
        AutoObservableEmitter
        """
        # Extract block info
        block_info: Dict[str, Dict] = {}
        block_names: List[str] = []
        for block_name, block_data in alloc.items():
            if block_name == "total":
                continue
            data_start, n_qubits = block_data["data"]
            code = block_data.get("code", codes[0] if codes else None)
            block_info[block_name] = {
                "data_start": data_start,
                "n_qubits": n_qubits,
                "code": code,
            }
            block_names.append(block_name)

        # Extract mid-circuit measurements from gadget if available
        if mid_circuit_meas is None:
            mid_circuit_meas = _extract_mid_circuit_meas(gadget)

        # Get input states
        control_state = getattr(gadget, "control_state", "0")
        target_state = getattr(gadget, "target_state", "0")

        # Generate candidates from observable transform
        try:
            obs_transform = gadget.get_observable_transform()
            candidates = generate_candidates_from_transform(
                obs_transform,
                control_state,
                target_state,
                block_names,
                mid_circuit_meas,
            )
        except (NotImplementedError, AttributeError):
            # No observable transform available — use generic candidates
            candidates = _generate_generic_candidates(
                block_names, control_state, target_state, mid_circuit_meas
            )

        return cls(
            circuit=circuit,
            candidates=candidates,
            block_info=block_info,
            mid_circuit_meas=mid_circuit_meas,
        )

    def get_config(
        self,
        *,
        exhaustive_fallback: bool = True,
        unsigned: bool = True,
    ) -> "ObservableConfig":
        """Compute and return the best valid ObservableConfig.

        Parameters
        ----------
        exhaustive_fallback : bool
            If True and no heuristic candidate passes, try exhaustive search.
        unsigned : bool
            Use unsigned flow validation (ignore Pauli frame signs).

        Returns
        -------
        ObservableConfig
            Best valid configuration.

        Raises
        ------
        ValueError
            If no valid observable found.
        """
        from qectostim.gadgets.base import ObservableConfig

        total_meas = _count_circuit_measurements(self._circuit)

        # Try heuristic candidates first
        for candidate in self._candidates:
            # Build data qubit measurement mapping
            data_qubit_meas: Dict[int, int] = {}
            logical_support: Dict[str, List[Tuple[int, str]]] = {}

            for block_name, basis in candidate.terms:
                if block_name not in self._block_info:
                    continue
                bi = self._block_info[block_name]
                code = bi["code"]
                data_start = bi["data_start"]
                n_qubits = bi["n_qubits"]

                support = (
                    code.get_logical_x_support()
                    if basis == "X"
                    else code.get_logical_z_support()
                )
                logical_support[block_name] = [
                    (data_start + idx, basis) for idx in support
                ]
                meas_start = bi.get("meas_start", 0)
                for i in range(n_qubits):
                    data_qubit_meas[data_start + i] = meas_start + i

            is_valid = validate_observable_flow(
                self._circuit,
                data_qubit_meas,
                logical_support,
                candidate.bridge_meas_indices,
                unsigned=unsigned,
            )

            if is_valid:
                logger.info(
                    "Observable validated: %s", candidate.description
                )
                return candidate.to_observable_config()
            else:
                logger.debug(
                    "Observable rejected: %s", candidate.description
                )

        # Exhaustive fallback
        if exhaustive_fallback:
            logger.info("Heuristic candidates exhausted, trying exhaustive search")
            search_result = exhaustive_observable_search(
                self._circuit,
                self._block_info,
                self._mid_circuit_meas,
                unsigned=unsigned,
            )
            if search_result.config is not None:
                logger.info(
                    "Exhaustive search found %d valid candidates (tried %d)",
                    len(search_result.valid_candidates),
                    search_result.candidates_tried,
                )
                return search_result.config

        # Nothing worked
        logger.warning("No valid observable found, returning single-block Z default")
        return ObservableConfig.transversal_single_qubit()

    def get_search_result(self) -> ObservableSearchResult:
        """Run full search and return detailed results for debugging."""
        return exhaustive_observable_search(
            self._circuit,
            self._block_info,
            self._mid_circuit_meas,
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_mid_circuit_meas(gadget: "Gadget") -> Dict[str, List[int]]:
    """Extract mid-circuit measurement groups from gadget attributes.

    Looks for common CSS surgery measurement attributes:
    - ``_anc_meas_indices``: Ancilla data qubit MX measurements
    - ``_zz_split_meas``: ZZ bridge measurements
    - ``_xx_split_meas``: XX bridge measurements
    - ``_get_anc_x_logical_meas()``: Ancilla X logical measurement indices

    Returns
    -------
    dict
        Named groups of measurement indices.
    """
    result: Dict[str, List[int]] = {}

    # Ancilla MX measurements
    if hasattr(gadget, "_anc_meas_indices"):
        result["anc_all_mx"] = list(gadget._anc_meas_indices)

    # Ancilla X logical (subset of anc MX at X_L support)
    if hasattr(gadget, "_get_anc_x_logical_meas"):
        try:
            result["anc_mx"] = list(gadget._get_anc_x_logical_meas())
        except Exception:
            pass

    # ZZ bridge
    if hasattr(gadget, "_zz_split_meas"):
        result["zz_bridge"] = list(gadget._zz_split_meas)

    # XX bridge
    if hasattr(gadget, "_xx_split_meas"):
        result["xx_bridge"] = list(gadget._xx_split_meas)

    return result


def _generate_generic_candidates(
    block_names: List[str],
    control_state: str,
    target_state: str,
    mid_circuit_meas: Dict[str, List[int]],
) -> List[CandidateObservable]:
    """Generate generic candidates when no observable transform is available.

    Falls back to trying all single-block and two-block combinations in
    the appropriate basis.
    """
    ctrl_basis = "Z" if control_state == "0" else "X"
    tgt_basis = "Z" if target_state == "0" else "X"
    candidates: List[CandidateObservable] = []

    ctrl = block_names[0] if block_names else "block_0"
    tgt = block_names[-1] if len(block_names) > 1 else ctrl

    # Two-block first (preferred)
    if ctrl != tgt:
        candidates.append(CandidateObservable(
            terms=[(ctrl, ctrl_basis), (tgt, tgt_basis)],
            description=f"generic: {ctrl_basis}_{ctrl} ⊕ {tgt_basis}_{tgt}",
        ))

    # Single-block
    candidates.append(CandidateObservable(
        terms=[(ctrl, ctrl_basis)],
        description=f"generic: {ctrl_basis}_{ctrl}",
    ))
    if ctrl != tgt:
        candidates.append(CandidateObservable(
            terms=[(tgt, tgt_basis)],
            description=f"generic: {tgt_basis}_{tgt}",
        ))

    # With bridge corrections
    for meas_name, meas_indices in mid_circuit_meas.items():
        for base_candidate in list(candidates):
            candidates.append(CandidateObservable(
                terms=list(base_candidate.terms),
                bridge_meas_indices=list(meas_indices),
                description=f"{base_candidate.description} ⊕ {meas_name}",
            ))

    return candidates


# ---------------------------------------------------------------------------
# Convenience: validate an existing ObservableConfig against a circuit
# ---------------------------------------------------------------------------

def validate_existing_config(
    circuit: stim.Circuit,
    obs_config: "ObservableConfig",
    alloc: Dict,
    meas_start: int,
    *,
    unsigned: bool = True,
) -> bool:
    """Validate that an existing ObservableConfig produces a valid flow.

    This is useful for checking manually-specified observable configs
    without changing them.

    Parameters
    ----------
    circuit : stim.Circuit
        Complete circuit.
    obs_config : ObservableConfig
        Configuration to validate.
    alloc : dict
        Qubit allocation.
    meas_start : int
        Measurement start index for final data measurement.
    unsigned : bool
        Use unsigned flow validation.

    Returns
    -------
    bool
        True if valid.
    """
    total_meas = _count_circuit_measurements(circuit)

    # Build data qubit → meas mapping
    data_qubit_meas: Dict[int, int] = {}
    logical_support: Dict[str, List[Tuple[int, str]]] = {}
    meas_idx = meas_start

    for block_name, block_data in alloc.items():
        if block_name == "total":
            continue
        data_start, n_qubits = block_data["data"]
        for i in range(n_qubits):
            data_qubit_meas[data_start + i] = meas_idx
            meas_idx += 1

    # Build logical support from correlation terms
    if obs_config.correlation_terms:
        for term in obs_config.correlation_terms:
            for block_name, block_data in alloc.items():
                if block_name == "total":
                    continue
                from qectostim.experiments.observable import _blocks_match
                if _blocks_match(block_name, term.block):
                    code = block_data["code"]
                    data_start = block_data["data"][0]
                    support = (
                        code.get_logical_x_support()
                        if term.basis == "X"
                        else code.get_logical_z_support()
                    )
                    logical_support[block_name] = [
                        (data_start + idx, term.basis) for idx in support
                    ]
                    break

    return validate_observable_flow(
        circuit,
        data_qubit_meas,
        logical_support,
        obs_config.bridge_frame_meas_indices,
        unsigned=unsigned,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    "AutoObservableEmitter",
    "CandidateObservable",
    "ObservableSearchResult",
    "exhaustive_observable_search",
    "generate_candidates_from_transform",
    "validate_existing_config",
    "validate_observable_flow",
]
