"""
Detector tracking utilities for FT experiments.

This module provides utility functions and state tracking for detector
construction in FaultTolerantGadgetExperiment. It handles:

- Measurement record tracking (MeasurementRecord)
- Stabilizer support computation (get_stabilizer_supports)
- Stabilizer product computation for detector terms (compute_stabilizer_product)
- Detector coordinate computation (compute_detector_coords)
- Inter-gadget chain state (GadgetChainState)

Design Principles:
-----------------
1. Gadgets declare WHAT via config dataclasses (PreparationConfig, etc.)
2. scheduling.py decides WHEN (stabilizer ordering for determinism)
3. detector_emission.py handles HOW to emit DETECTOR instructions
4. This module provides UTILITIES that the above consume

Rules for Deterministic Detectors:
----------------------------------
ANCHORS (first round, after preparation):
  - |0⟩ prep → Z stabilizers deterministic → Z anchors
  - |+⟩ prep → X stabilizers deterministic → X anchors

BOUNDARIES (last round, before destructive measurement):
  - MZ measurement → Z stabilizers correlate with data → Z boundaries
  - MX measurement → X stabilizers correlate with data → X boundaries

TEMPORALS (consecutive rounds):
  - Always 2-term: syndrome(round N) ⊕ syndrome(round N+1)

CROSSINGS (around transversal gates):
  - Formulas from CrossingDetectorConfig (2-term or 3-term)
"""

from typing import Dict, List, Optional, Tuple, Set, Any, TYPE_CHECKING
from dataclasses import dataclass, field
import stim

from ..gadgets.base import (
    CrossingDetectorConfig,
    CrossingDetectorFormula,
    CrossingDetectorTerm,
)

if TYPE_CHECKING:
    from ..codes.abstract_code import Code


# =============================================================================
# Measurement record tracking
# =============================================================================

@dataclass
class MeasurementRecord:
    """
    Tracks measurement indices for detector construction.
    
    Maintains the mapping between logical measurements and their
    indices in the Stim measurement record, organized by block,
    stabilizer type, and timing.
    """
    records: Dict[str, List[int]] = field(default_factory=dict)
    
    def get_indices(self, block: str, stab_type: str, timing: str) -> List[int]:
        """Get measurement indices for a specific stabilizer measurement."""
        key = f"{block}_{stab_type}_{timing}"
        return self.records.get(key, [])
    
    def add_measurement(self, block: str, stab_type: str, timing: str, indices: List[int]):
        """Add measurement indices to the record."""
        key = f"{block}_{stab_type}_{timing}"
        self.records[key] = indices



# =============================================================================
# Helper functions for stabilizer support and product computation
# =============================================================================

def get_stabilizer_supports(code: "Code", stabilizer_type: str) -> List[List[int]]:
    """
    Get the qubit support for each stabilizer of the given type.
    
    Uses the Code ABC interface directly (get_x_stabilizers/get_z_stabilizers).
    
    Parameters
    ----------
    code : Code
        The code object.
    stabilizer_type : str
        "X" or "Z".
        
    Returns
    -------
    List[List[int]]
        List of qubit indices for each stabilizer.
    """
    if stabilizer_type.upper() == "Z":
        return code.get_z_stabilizers()
    else:
        return code.get_x_stabilizers()


def compute_stabilizer_product(
    data_meas_indices: List[int],
    stabilizer_support: List[int],
    total_measurements: int,
) -> List[stim.GateTarget]:
    """
    Compute the XOR product of data measurements for a stabilizer.
    
    For a stabilizer with support on qubits {i, j, k}, this returns
    rec[] targets for data_meas[i] ⊕ data_meas[j] ⊕ data_meas[k].
    """
    targets = []
    for qubit_idx in stabilizer_support:
        if qubit_idx < len(data_meas_indices):
            abs_idx = data_meas_indices[qubit_idx]
            offset = abs_idx - total_measurements
            targets.append(stim.target_rec(offset))
    return targets


# =============================================================================
# Detector coordinate computation
# =============================================================================

def compute_detector_coords(
    block_name: str,
    stabilizer_type: str,
    stabilizer_index: int,
    time_coord: float,
    block_offset: Tuple[float, ...] = (0.0, 0.0),
) -> Tuple[float, ...]:
    """
    Compute detector coordinates for a stabilizer detector.
    """
    type_offset = 0.5 if stabilizer_type == "X" else 0.0
    x = block_offset[0] + stabilizer_index + type_offset
    y = block_offset[1] if len(block_offset) > 1 else 0.0
    return (x, y, time_coord)


# =============================================================================
# Gadget chain state (for multi-gadget circuits)
# =============================================================================

@dataclass
class GadgetChainState:
    """
    Tracks state needed for inter-gadget detector emission in a chain.
    
    Captures the essential information from a completed gadget
    that's needed to emit inter-gadget crossing detectors when the next
    gadget begins.
    """
    final_syndrome_meas: Dict[str, Dict[str, List[int]]]
    outgoing_stabilizer_correlations: Optional[CrossingDetectorConfig] = None
    measurement_index: int = 0
    pauli_frame_state: Optional[Dict] = None
    
    def has_crossing_detectors(self) -> bool:
        """Check if this gadget output requires crossing detectors."""
        return (
            self.outgoing_stabilizer_correlations is not None and
            len(self.outgoing_stabilizer_correlations.formulas) > 0
        )
    
    def serialize(self) -> Dict[str, Any]:
        """Serialize for storage/transmission between processes."""
        return {
            "final_syndrome_meas": self.final_syndrome_meas,
            "outgoing_stabilizer_correlations": (
                self.outgoing_stabilizer_correlations.to_dict()
                if self.outgoing_stabilizer_correlations else None
            ),
            "measurement_index": self.measurement_index,
            "pauli_frame_state": self.pauli_frame_state,
        }
    
    @classmethod
    def deserialize(cls, data: Dict[str, Any]) -> "GadgetChainState":
        """Restore from serialized data."""
        crossing_config = None
        if data.get("outgoing_stabilizer_correlations"):
            config_data = data["outgoing_stabilizer_correlations"]
            formulas = []
            for formula_data in config_data.get("formulas", []):
                terms = [
                    CrossingDetectorTerm(
                        block=t["block"],
                        stabilizer_type=t["stabilizer_type"],
                        timing=t["timing"],
                    )
                    for t in formula_data.get("terms", [])
                ]
                formulas.append(CrossingDetectorFormula(terms=terms))
            crossing_config = CrossingDetectorConfig(formulas=formulas)
        
        return cls(
            final_syndrome_meas=data["final_syndrome_meas"],
            outgoing_stabilizer_correlations=crossing_config,
            measurement_index=data["measurement_index"],
            pauli_frame_state=data.get("pauli_frame_state"),
        )


# =============================================================================
# Detector Coverage Resolver
# =============================================================================

class DetectorCoverageResolver:
    """
    High-level resolver for detector coverage across all blocks.

    Given a gadget's config declarations (PreparationConfig, MeasurementConfig,
    CrossingDetectorConfig, StabilizerTransform), determines the complete
    detector coverage plan for both pre-gadget and post-gadget rounds.

    This consolidates the scheduling orchestration that was previously spread
    across _compute_block_schedules() and _compute_post_gadget_schedules()
    in ft_gadget_experiment.py.

    Usage::

        resolver = DetectorCoverageResolver(scheduler)
        pre_schedules = resolver.compute_pre_gadget_coverage(
            builders, prep_config, meas_basis, num_rounds, default_ordering,
        )
        post_schedules = resolver.compute_post_gadget_coverage(
            builders, prep_config, meas_basis, num_rounds,
            stab_transform, destroyed_blocks, default_ordering,
        )
    """

    def __init__(self, scheduler: "StabilizerScheduler"):
        self._scheduler = scheduler

    def compute_pre_gadget_coverage(
        self,
        builders: list,
        prep_config: "PreparationConfig",
        meas_basis: str,
        num_rounds: int,
        default_ordering: Optional[str] = None,
        block_meas_bases: Optional[Dict[str, str]] = None,
    ) -> Dict[str, "BlockSchedule"]:
        """
        Compute per-block schedules for pre-gadget stabilizer rounds.

        Parameters
        ----------
        builders : list
            Round builders for each code block.
        prep_config : PreparationConfig
            Gadget's preparation config (declares initial state per block).
        meas_basis : str
            Experiment measurement basis ("Z" or "X"). Used as fallback.
        num_rounds : int
            Number of pre-gadget rounds.
        default_ordering : Optional[str]
            Override for middle-round ordering ("Z_FIRST" or "X_FIRST").
        block_meas_bases : Optional[Dict[str, str]]
            Per-block measurement basis override. If provided, each block
            uses its own measurement basis for boundary scheduling.
            Falls back to meas_basis if block not in dict.

        Returns
        -------
        Dict[str, BlockSchedule]
            Per-block scheduling decisions.
        """
        schedules: Dict[str, "BlockSchedule"] = {}

        for builder in builders:
            prep_basis = "0"
            block_config = prep_config.get_block_config(builder.block_name)
            if block_config is not None:
                prep_basis = block_config.initial_state

            # Use per-block measurement basis if available, else global
            block_meas = meas_basis
            if block_meas_bases and builder.block_name in block_meas_bases:
                block_meas = block_meas_bases[builder.block_name]

            schedule = self._scheduler.compute_block_schedule(
                block_name=builder.block_name,
                prep_basis=prep_basis,
                meas_basis=block_meas,
                num_rounds=num_rounds,
                default_ordering=default_ordering,
            )
            schedules[builder.block_name] = schedule

        return schedules

    def compute_post_gadget_coverage(
        self,
        builders: list,
        prep_config: "PreparationConfig",
        meas_basis: str,
        num_rounds: int,
        stab_transform: "StabilizerTransform",
        default_ordering: Optional[str] = None,
        block_meas_bases: Optional[Dict[str, str]] = None,
    ) -> Dict[str, "BlockSchedule"]:
        """
        Compute per-block schedules for post-gadget rounds on surviving blocks.

        Unlike pre-gadget, post-gadget rounds must account for the
        StabilizerTransform's swap_xz flag (e.g., H gate swaps X↔Z).

        Parameters
        ----------
        builders : list
            Round builders for surviving code blocks only.
        prep_config : PreparationConfig
            Gadget's preparation config.
        meas_basis : str
            Experiment measurement basis. Used as fallback.
        num_rounds : int
            Number of post-gadget rounds.
        stab_transform : StabilizerTransform
            The gadget's stabilizer transformation.
        default_ordering : Optional[str]
            Override for middle-round ordering.
        block_meas_bases : Optional[Dict[str, str]]
            Per-block measurement basis override. Falls back to meas_basis
            if block not in dict.

        Returns
        -------
        Dict[str, BlockSchedule]
            Per-block scheduling decisions.
        """
        schedules: Dict[str, "BlockSchedule"] = {}

        for builder in builders:
            prep_basis = "0"
            block_config = prep_config.get_block_config(builder.block_name)
            if block_config is not None:
                prep_basis = block_config.initial_state

            # Apply swap_xz transformation
            # For H gate: |0⟩ (Z det.) → effective |+⟩ (X det.)
            #              |+⟩ (X det.) → effective |0⟩ (Z det.)
            effective_prep = prep_basis
            if stab_transform.swap_xz:
                effective_prep = "+" if prep_basis == "0" else "0"

            # Use per-block measurement basis if available, else global
            block_meas = meas_basis
            if block_meas_bases and builder.block_name in block_meas_bases:
                block_meas = block_meas_bases[builder.block_name]

            schedule = self._scheduler.compute_block_schedule(
                block_name=builder.block_name,
                prep_basis=effective_prep,
                meas_basis=block_meas,
                num_rounds=num_rounds,
                default_ordering=default_ordering,
            )
            schedules[builder.block_name] = schedule

        return schedules

