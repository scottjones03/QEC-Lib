"""
State preparation module for FT experiments.

This module handles state preparation logic for FaultTolerantGadgetExperiment:
- Initial state preparation (R for |0⟩, RX for |+⟩)
- Deterministic stabilizer queries for anchor detection

Design Principles:
-----------------
1. Gadgets declare WHAT via PreparationConfig
2. This module implements HOW to prepare states
3. Experiment orchestrates by passing configs to this module
4. No type-checking branching anywhere
"""

from typing import Dict, List, TYPE_CHECKING
import stim

if TYPE_CHECKING:
    from qectostim.gadgets.base import PreparationConfig


def get_qubits_to_skip_reset(
    prep_config: "PreparationConfig",
    block_data_qubits: Dict[str, List[int]],
    use_rx: bool = True,
) -> set:
    """
    Get qubits that should skip the initial reset.
    
    BEHAVIOR DEPENDS ON use_rx:
    
    When use_rx=True (default):
        RX handles reset atomically, so no qubits need to skip reset.
        Data qubits will be reset normally, then RX applied for |+⟩ blocks.
        
    When use_rx=False (legacy):
        For |+⟩ blocks, we rely on Stim's implicit |0⟩ → apply H → |+⟩.
        These qubits should NOT be reset before H because:
        - If we do R then H, Stim traces X stabilizers backward: X → (H) → Z → (R) 
        - The R on |0⟩ with Z Pauli is non-deterministic
        - If we do only H (no R), the trace is: X → (H) → Z on implicit |0⟩ = +1
    
    Parameters
    ----------
    prep_config : PreparationConfig
        Per-block preparation configuration from gadget.
    block_data_qubits : Dict[str, List[int]]
        Mapping from block name to data qubit indices.
    use_rx : bool
        If True, using RX for |+⟩ (no skip needed). If False, legacy H mode.
        
    Returns
    -------
    set
        Set of qubit indices that should skip the initial R (reset).
    """
    # RX mode: no skipping needed, RX is atomic
    if use_rx:
        return set()
    
    # Legacy H mode: skip reset for qubits that will get H
    skip_reset = set()
    
    # Try both the provided block names AND any semantic block configs
    for block_name, qubits in block_data_qubits.items():
        # Use get_block_config for name normalization
        # This handles block_0 → data_block mapping
        block_config = prep_config.get_block_config(block_name)
        if block_config is None:
            continue
            
        # Skip reset for qubits that will get H for |+⟩ preparation
        # These qubits rely on Stim's implicit |0⟩ → apply H → |+⟩
        if block_config.initial_state in ("+", "-"):
            if not block_config.skip_experiment_prep:
                skip_reset.update(qubits)
    
    return skip_reset


def emit_initial_states(
    circuit: stim.Circuit,
    prep_config: "PreparationConfig",
    block_data_qubits: Dict[str, List[int]],
    use_rx: bool = True,
) -> None:
    """
    Emit initial state preparation for all blocks.
    
    For |0⟩: Just R (reset) - already done by qubit initialization
    For |+⟩: Use RX if use_rx=True, otherwise H (legacy mode)
    
    GATE OPTIMIZATION:
    ------------------
    We prefer RX over H for |+⟩ preparation because:
    - RX is a single instruction: RX ≡ prepare in X-basis eigenstate
    - H requires Stim's implicit |0⟩ → H → |+⟩ (extra gate tracking)
    - RX produces cleaner detector backward-tracking
    
    NOTE: When use_rx=True, we do NOT skip reset - RX handles both
    reset AND preparation atomically. When use_rx=False (legacy),
    we skip reset and apply H (relies on Stim's implicit |0⟩).
    
    Parameters
    ----------
    circuit : stim.Circuit
        Circuit to emit into.
    prep_config : PreparationConfig
        Per-block preparation configuration from gadget.
    block_data_qubits : Dict[str, List[int]]
        Mapping from block name to data qubit indices.
    use_rx : bool
        If True, use RX for |+⟩ prep. If False, use H (legacy).
    """
    from ..gadgets.scheduling import StabilizerScheduler
    scheduler = StabilizerScheduler()
    
    # Iterate over the provided block names and use normalization
    for block_name, data_qubits in block_data_qubits.items():
        # Use get_block_config for name normalization (block_0 → data_block)
        block_config = prep_config.get_block_config(block_name)
        if block_config is None:
            continue
        
        # |+⟩ or |-⟩ requires basis preparation
        if block_config.initial_state in ("+", "-"):
            if use_rx:
                # Use RX: atomic prepare-in-X-basis
                # No need to skip reset since RX handles it
                circuit.append("RX", data_qubits)
            else:
                # Legacy mode: H after implicit |0⟩ 
                circuit.append("H", data_qubits)
    
    circuit.append("TICK")


def get_deterministic_stabilizers(
    prep_config: "PreparationConfig",
    block_name: str,
) -> Dict[str, bool]:
    """
    Get which stabilizer types are deterministic for a block.
    
    Uses block name normalization to handle block_0 → data_block mapping.
    
    Parameters
    ----------
    prep_config : PreparationConfig
        Preparation configuration.
    block_name : str
        Name of the block to query (can be "block_0" or "data_block").
        
    Returns
    -------
    Dict[str, bool]
        {"Z": z_deterministic, "X": x_deterministic}
    """
    # Use get_block_config for name normalization
    block_config = prep_config.get_block_config(block_name)
    if block_config is None:
        return {"Z": False, "X": False}
    
    return {
        "Z": block_config.z_deterministic,
        "X": block_config.x_deterministic,
    }
