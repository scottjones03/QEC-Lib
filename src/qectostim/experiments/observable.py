"""
Observable emission module for FT experiments.

This module is the SINGLE SOURCE OF TRUTH for all observable logic in
FaultTolerantGadgetExperiment. It handles:

- Single-block observables (e.g., Z_L(A) for transversal gates)
- Two-block correlation observables (e.g., X_L(D) ⊕ Z_L(A) for CZ |+⟩)
- Two-qubit gate transforms (CNOT Z→Z⊗Z, X⊗I→X)
- Frame corrections from teleportation (XOR'd into observable)
- Raw sampling metadata setup (skip OBSERVABLE_INCLUDE)
- Hybrid decoding (emit clean obs per block, corrections applied classically)

Design Principles:
-----------------
1. Gadgets declare WHAT via ObservableConfig
2. This module implements HOW to emit OBSERVABLE_INCLUDE
3. Experiment calls emit_observable() with alloc + ctx → done
4. No gadget type checking anywhere
5. All branching driven by ObservableConfig fields, NOT by gadget identity
"""

from typing import Dict, List, Optional, Any, Set, TYPE_CHECKING
from dataclasses import dataclass, field

import numpy as np
import stim

if TYPE_CHECKING:
    from qectostim.gadgets.base import ObservableConfig, Gadget
    from qectostim.codes.abstract_code import Code
    from qectostim.experiments.stabilizer_rounds import DetectorContext


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_logical_support(code: "Code", basis: str, idx: int = 0) -> List[int]:
    """
    Get the support (qubit indices) of a logical operator.
    
    Uses the standard Code interface: code.get_logical_x_support() /
    code.get_logical_z_support() defined in abstract_code.py.
    No hasattr probing — all Code subclasses implement these methods.
    
    Parameters
    ----------
    code : Code
        The code object (must implement Code ABC).
    basis : str
        "X" for X logical, "Z" for Z logical.
    idx : int
        Index of logical qubit (default 0).
        
    Returns
    -------
    List[int]
        Qubit indices in the support of the logical operator.
    """
    if basis.upper() == "Z":
        return code.get_logical_z_support(idx)
    else:
        return code.get_logical_x_support(idx)


def _build_qubit_to_meas(
    alloc: Dict[str, Any],
    meas_start: int,
    destroyed_block_meas_starts: Optional[Dict[str, int]] = None,
) -> Dict[int, int]:
    """Build a mapping from global qubit index → measurement index.
    
    Parameters
    ----------
    alloc : Dict[str, Any]
        Qubit allocation dict (block_name → {"data": (start, n), "code": ...}).
        May include both surviving AND destroyed blocks.
    meas_start : int
        First measurement index of the data-qubit measurement round
        (for surviving blocks).
    destroyed_block_meas_starts : Optional[Dict[str, int]]
        Measurement start indices for destroyed blocks (from gadget-internal
        measurements). Keys are block names, values are absolute measurement
        indices where the block's data qubits were measured.
    
    Returns
    -------
    Dict[int, int]
        {global_qubit_idx: measurement_idx}
    """
    if destroyed_block_meas_starts is None:
        destroyed_block_meas_starts = {}
    
    qubit_to_meas: Dict[int, int] = {}
    meas_idx = meas_start
    for block_name, block_info in alloc.items():
        if block_name == "total":
            continue
        data_start, n = block_info["data"]
        if block_name in destroyed_block_meas_starts:
            # Destroyed block: use gadget-internal measurement start
            dest_meas_idx = destroyed_block_meas_starts[block_name]
            for i in range(n):
                qubit_to_meas[data_start + i] = dest_meas_idx + i
        else:
            # Surviving block: use final measurement start
            for i in range(n):
                qubit_to_meas[data_start + i] = meas_idx
                meas_idx += 1
    return qubit_to_meas


def _collect_logical_meas(
    code: "Code",
    data_start: int,
    basis: str,
    qubit_to_meas: Dict[int, int],
) -> List[int]:
    """Collect measurement indices for a logical operator.
    
    Parameters
    ----------
    code : Code
        Code object for the block.
    data_start : int
        Starting global qubit index for the block's data qubits.
    basis : str
        "X" or "Z".
    qubit_to_meas : Dict[int, int]
        qubit → measurement index mapping.
    
    Returns
    -------
    List[int]
        Measurement indices corresponding to the logical support.
    """
    support = get_logical_support(code, basis, 0)
    meas_indices = []
    for local_idx in support:
        global_idx = data_start + local_idx
        if global_idx in qubit_to_meas:
            meas_indices.append(qubit_to_meas[global_idx])
    return meas_indices


def _emit_targets(
    circuit: stim.Circuit,
    meas_indices: List[int],
    total_measurements: int,
    obs_idx: int = 0,
) -> None:
    """Emit OBSERVABLE_INCLUDE for a list of measurement indices.
    
    Parameters
    ----------
    circuit : stim.Circuit
        Circuit to append to.
    meas_indices : List[int]
        Absolute measurement indices.
    total_measurements : int
        Current total measurement count (for rec[] offset).
    obs_idx : int
        Observable index.
    """
    if meas_indices:
        lookbacks = [idx - total_measurements for idx in meas_indices]
        targets = [stim.target_rec(lb) for lb in lookbacks]
        circuit.append("OBSERVABLE_INCLUDE", targets, obs_idx)


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------

@dataclass
class RawSamplingMetadata:
    """Metadata for raw-sampling decoder path (no OBSERVABLE_INCLUDE emitted).
    
    When ObservableConfig.requires_raw_sampling is True, the gadget needs
    classical post-processing of raw measurement outcomes. This dataclass
    stores the information needed by the decoder.
    """
    output_block: str
    output_code: Optional["Code"]
    output_data_start: Optional[int]
    measurement_basis: str
    meas_start: int
    logical_support: List[int]
    num_data_qubits: int


@dataclass
class HybridDecodingMetadata:
    """Metadata for hybrid DEM + classical frame tracking.
    
    For teleportation with hybrid decoding, two clean observables are emitted
    (without frame corrections), and frame corrections are applied classically.
    """
    data_code: Optional["Code"]
    ancilla_code: Optional["Code"]
    data_data_start: Optional[int]
    ancilla_data_start: Optional[int]
    meas_start: int
    qubit_to_meas: Dict[int, int]


# ---------------------------------------------------------------------------
# Core entry point: blocks_match import
# ---------------------------------------------------------------------------

def _blocks_match(name: str, target: str) -> bool:
    """Check if a block name matches a target, handling aliasing.
    
    Delegates to gadgets.base.blocks_match() for canonical matching.
    This avoids coupling to the full base module at import time.
    """
    from qectostim.gadgets.base import blocks_match, normalize_block_name
    return blocks_match(name, target)


# ---------------------------------------------------------------------------
# Primary entry point
# ---------------------------------------------------------------------------

def emit_observable(
    circuit: stim.Circuit,
    obs_config: "ObservableConfig",
    alloc: Dict[str, Any],
    ctx: "DetectorContext",
    meas_start: int,
    measurement_basis: str,
    gadget: "Gadget",
    destroyed_block_meas_starts: Optional[Dict[str, int]] = None,
) -> Optional[Any]:
    """
    Emit OBSERVABLE_INCLUDE from ObservableConfig — the ONE function to call.
    
    This dispatches on ObservableConfig fields (NOT gadget type) to handle
    all observable paths:
    
    1. requires_raw_sampling → store metadata, skip OBSERVABLE_INCLUDE
    2. use_hybrid_decoding → emit two clean observables (data + ancilla)
    3. correlation_terms → multi-block XOR (e.g. X_L(D) ⊕ Z_L(A))
    4. two_qubit_transform → CNOT/CZ observable transforms
    5. output_blocks → standard single/multi-block observables
    
    Frame corrections from config.frame_correction_blocks are XOR'd in
    for paths 3-5.
    
    Parameters
    ----------
    circuit : stim.Circuit
        Circuit to emit into.
    obs_config : ObservableConfig
        Observable configuration from gadget.get_observable_config().
    alloc : Dict[str, Any]
        Qubit allocation dict. Should include ALL blocks (surviving + destroyed)
        so that correlation terms referencing destroyed blocks can be resolved.
    ctx : DetectorContext
        Detector context for measurement tracking.
    meas_start : int
        Starting measurement index for data qubit measurements.
    measurement_basis : str
        Experiment measurement basis ("X" or "Z").
    gadget : Gadget
        Gadget instance (used ONLY for block name queries via ABC methods,
        NOT for isinstance checks).
    destroyed_block_meas_starts : Optional[Dict[str, int]]
        Measurement start indices for destroyed blocks. When a block is
        destroyed during gadget execution (e.g., data_block in teleportation),
        its measurements are emitted by the gadget at a different point.
        This dict maps block names to the absolute measurement index where
        the block's data qubits were measured.
    
    Returns
    -------
    Optional[Any]
        - RawSamplingMetadata if raw sampling path
        - HybridDecodingMetadata if hybrid decoding path
        - None for standard paths (OBSERVABLE_INCLUDE already emitted)
    """
    # Path 1: Raw sampling — skip OBSERVABLE_INCLUDE entirely
    if obs_config.requires_raw_sampling:
        return _setup_raw_sampling_metadata(
            alloc, meas_start, measurement_basis, gadget
        )
    
    # Path 2: Hybrid decoding — two clean observables
    if obs_config.use_hybrid_decoding:
        return _emit_hybrid_observables(
            circuit, alloc, ctx, meas_start, gadget
        )
    
    # Build qubit→measurement mapping for paths 3-5
    qubit_to_meas = _build_qubit_to_meas(alloc, meas_start, destroyed_block_meas_starts)
    observable_meas: List[int] = []
    
    # Path 3: Correlation terms (e.g. X_L(D) ⊕ Z_L(A))
    if obs_config.correlation_terms:
        for term in obs_config.correlation_terms:
            for block_name, block_info in alloc.items():
                if block_name == "total":
                    continue
                if _blocks_match(block_name, term.block):
                    code = block_info["code"]
                    data_start, _ = block_info["data"]
                    observable_meas.extend(
                        _collect_logical_meas(code, data_start, term.basis, qubit_to_meas)
                    )
                    break
    
    # Path 4: Two-qubit transform (CNOT/CZ observable transforms)
    elif obs_config.two_qubit_transform is not None:
        observable_meas = _collect_two_qubit_transform_meas(
            obs_config.two_qubit_transform, alloc, qubit_to_meas, measurement_basis
        )
    
    # Path 5: Standard output blocks
    elif obs_config.output_blocks:
        observable_meas = _collect_output_block_meas(
            obs_config, alloc, qubit_to_meas, measurement_basis
        )
    
    # Frame corrections (paths 3-5)
    if obs_config.frame_correction_blocks:
        for block_name in alloc:
            if block_name == "total":
                continue
            frame_meas = ctx.get_frame_correction_measurements(
                block_name, measurement_basis
            )
            observable_meas.extend(frame_meas)
    
    # Emit the observable
    _emit_targets(circuit, observable_meas, ctx.measurement_index, obs_idx=0)
    return None


# ---------------------------------------------------------------------------
# Path helpers (private)
# ---------------------------------------------------------------------------

def _collect_two_qubit_transform_meas(
    transform: Any,
    alloc: Dict[str, Any],
    qubit_to_meas: Dict[int, int],
    measurement_basis: str,
) -> List[int]:
    """Collect measurement indices for two-qubit gate observable transforms.
    
    For CNOT/CZ, the observable transforms across blocks:
    - Z measurement: use transform.control_z_to
    - X measurement: use transform.control_x_to
    
    Parameters
    ----------
    transform : TwoQubitObservableTransform
        Describes how observables transform.
    alloc : Dict[str, Any]
        Qubit allocation.
    qubit_to_meas : Dict[int, int]
        Qubit → measurement index mapping.
    measurement_basis : str
        "X" or "Z".
    
    Returns
    -------
    List[int]
        Measurement indices.
    """
    from qectostim.gadgets.base import normalize_block_name
    
    if measurement_basis.upper() == "Z":
        ctrl_to = transform.control_z_to
    else:
        ctrl_to = transform.control_x_to
    
    blocks_bases: Dict[str, str] = {}
    if ctrl_to[0] is not None:
        blocks_bases["block_0"] = ctrl_to[0]
    if ctrl_to[1] is not None:
        blocks_bases["block_1"] = ctrl_to[1]
    
    meas_indices: List[int] = []
    for block_name, block_info in alloc.items():
        if block_name == "total":
            continue
        canonical = normalize_block_name(block_name)
        if canonical not in blocks_bases:
            continue
        code = block_info["code"]
        data_start, _ = block_info["data"]
        basis = blocks_bases[canonical]
        meas_indices.extend(
            _collect_logical_meas(code, data_start, basis, qubit_to_meas)
        )
    return meas_indices


def _collect_output_block_meas(
    obs_config: "ObservableConfig",
    alloc: Dict[str, Any],
    qubit_to_meas: Dict[int, int],
    measurement_basis: str,
) -> List[int]:
    """Collect measurement indices from output blocks.
    
    Parameters
    ----------
    obs_config : ObservableConfig
        Observable configuration.
    alloc : Dict[str, Any]
        Qubit allocation.
    qubit_to_meas : Dict[int, int]
        Qubit → measurement index mapping.
    measurement_basis : str
        Default measurement basis.
    
    Returns
    -------
    List[int]
        Measurement indices.
    """
    meas_indices: List[int] = []
    for output_block in obs_config.output_blocks:
        for block_name, block_info in alloc.items():
            if block_name == "total":
                continue
            if _blocks_match(block_name, output_block):
                code = block_info["code"]
                data_start, _ = block_info["data"]
                # Use block-specific basis or default to measurement_basis
                basis = obs_config.block_bases.get(
                    block_name,
                    obs_config.block_bases.get(output_block, measurement_basis),
                )
                meas_indices.extend(
                    _collect_logical_meas(code, data_start, basis, qubit_to_meas)
                )
                break
    return meas_indices


def _setup_raw_sampling_metadata(
    alloc: Dict[str, Any],
    meas_start: int,
    measurement_basis: str,
    gadget: "Gadget",
) -> RawSamplingMetadata:
    """Set up metadata for raw-sampling decoder path.
    
    No OBSERVABLE_INCLUDE is emitted. The returned metadata tells the
    decoder which measurements correspond to the logical output.
    
    Parameters
    ----------
    alloc : Dict[str, Any]
        Qubit allocation.
    meas_start : int
        Starting measurement index.
    measurement_basis : str
        "X" or "Z".
    gadget : Gadget
        Gadget instance (for get_output_block_name()).
    
    Returns
    -------
    RawSamplingMetadata
    """
    output_block_name = gadget.get_output_block_name()
    output_code = None
    output_data_start = None
    
    for block_name, block_info in alloc.items():
        if block_name == "total":
            continue
        if _blocks_match(block_name, output_block_name):
            output_code = block_info["code"]
            output_data_start, _ = block_info["data"]
            break
    
    logical_support = []
    if output_code is not None:
        logical_support = get_logical_support(output_code, measurement_basis, 0)
    
    return RawSamplingMetadata(
        output_block=output_block_name,
        output_code=output_code,
        output_data_start=output_data_start,
        measurement_basis=measurement_basis,
        meas_start=meas_start,
        logical_support=logical_support,
        num_data_qubits=output_code.n if output_code else 0,
    )


def _emit_hybrid_observables(
    circuit: stim.Circuit,
    alloc: Dict[str, Any],
    ctx: "DetectorContext",
    meas_start: int,
    gadget: "Gadget",
) -> HybridDecodingMetadata:
    """
    Emit TWO clean observables for hybrid DEM + classical frame tracking.
    
    For teleportation with hybrid decoding:
    - OBSERVABLE_0: Data block Z_L measurements (teleportation outcome)
    - OBSERVABLE_1: Ancilla block Z_L measurements (final logical state)
      plus X stabilizer prep measurements with odd X_L overlap
    
    These observables do NOT include frame correction measurements.
    Frame corrections are applied classically after decoding:
    
      Final = (decoded OBS_1) XOR (projection_frame) XOR (decoded OBS_0)
    
    Parameters
    ----------
    circuit : stim.Circuit
        Circuit to emit into.
    alloc : Dict[str, Any]
        Qubit allocation.
    ctx : DetectorContext
        For measurement_index (rec[] offset calculation).
    meas_start : int
        Starting measurement index for data qubit measurements.
    gadget : Gadget
        Gadget instance (for get_input_block_name/get_output_block_name).
    
    Returns
    -------
    HybridDecodingMetadata
    """
    qubit_to_meas = _build_qubit_to_meas(alloc, meas_start)
    
    # ---- Data block (OBSERVABLE_0): Z_L teleportation measurement ----
    data_block_name = gadget.get_input_block_name()
    data_code = None
    data_data_start = None
    for block_name, block_info in alloc.items():
        if block_name == "total":
            continue
        if _blocks_match(block_name, data_block_name):
            data_code = block_info["code"]
            data_data_start, _ = block_info["data"]
            break
    
    if data_code is not None:
        obs0_meas = _collect_logical_meas(
            data_code, data_data_start, "Z", qubit_to_meas
        )
        _emit_targets(circuit, obs0_meas, ctx.measurement_index, obs_idx=0)
    
    # ---- Ancilla block (OBSERVABLE_1): Z_L final + X stab corrections ----
    ancilla_block_name = gadget.get_output_block_name()
    ancilla_code = None
    ancilla_data_start = None
    for block_name, block_info in alloc.items():
        if block_name == "total":
            continue
        if _blocks_match(block_name, ancilla_block_name):
            ancilla_code = block_info["code"]
            ancilla_data_start, _ = block_info["data"]
            break
    
    if ancilla_code is not None:
        # Z_L on ancilla (after H: original |+⟩_L → |0⟩_L)
        obs1_meas = _collect_logical_meas(
            ancilla_code, ancilla_data_start, "Z", qubit_to_meas
        )
        
        # Include X stabilizer prep measurements with odd X_L overlap
        # These absorb backward X sensitivity from tracking through H
        x_ref_meas: List[int] = []
        prep_meas_start = gadget.get_prep_meas_start()
        if prep_meas_start is not None:
            x_support = set(get_logical_support(ancilla_code, "X", 0))
            if ancilla_code.hx is not None:
                for stab_idx in range(ancilla_code.hx.shape[0]):
                    stab_support = set(np.where(ancilla_code.hx[stab_idx])[0])
                    overlap = len(stab_support & x_support)
                    if overlap % 2 == 1:  # Odd overlap absorbs X_L sensitivity
                        x_ref_meas.append(prep_meas_start + stab_idx)
        
        all_obs1 = obs1_meas + x_ref_meas
        _emit_targets(circuit, all_obs1, ctx.measurement_index, obs_idx=1)
    
    return HybridDecodingMetadata(
        data_code=data_code,
        ancilla_code=ancilla_code,
        data_data_start=data_data_start,
        ancilla_data_start=ancilla_data_start,
        meas_start=meas_start,
        qubit_to_meas=qubit_to_meas.copy(),
    )
