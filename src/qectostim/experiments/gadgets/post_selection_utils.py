"""
Post-selection utilities for enforcing fault-tolerant ancilla verification.

These helpers extract verification failures from Stim sampling results and
filter shots that don't pass verification checks.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, List, Dict, Optional, Set
import numpy as np

if TYPE_CHECKING:
    from .base import MeasurementMap


def extract_verification_failures(
    samples: np.ndarray,
    mmap: "MeasurementMap",
) -> np.ndarray:
    """
    Extract shot indices where verification checks failed.
    
    Parameters
    ----------
    samples : np.ndarray
        Measurement samples from Stim, shape (n_shots, n_measurements).
    mmap : MeasurementMap
        Measurement map containing verification_detectors.
    
    Returns
    -------
    np.ndarray
        Boolean array of shape (n_shots,) where True indicates verification failure.
    """
    n_shots = samples.shape[0]
    failures = np.zeros(n_shots, dtype=bool)
    
    # Check all verification measurements across all blocks
    for block_id, verif_meas in mmap.verification_measurements.items():
        for meas_idx in verif_meas:
            # Adjust for measurement offset
            adjusted_idx = meas_idx - mmap.offset
            if 0 <= adjusted_idx < samples.shape[1]:
                # Any non-zero verification measurement indicates failure
                failures |= (samples[:, adjusted_idx] != 0)
    
    return failures


def filter_shots(
    samples: np.ndarray,
    mmap: "MeasurementMap",
    keep_passing: bool = True,
) -> np.ndarray:
    """
    Filter shots based on verification outcomes.
    
    Parameters
    ----------
    samples : np.ndarray
        Measurement samples from Stim, shape (n_shots, n_measurements).
    mmap : MeasurementMap
        Measurement map containing verification info.
    keep_passing : bool
        If True, keep only passing shots. If False, keep only failing shots.
    
    Returns
    -------
    np.ndarray
        Filtered samples, shape (n_passing_shots, n_measurements) or
        (n_failing_shots, n_measurements).
    """
    if not mmap.verification_measurements:
        # No verification to filter on
        return samples
    
    failures = extract_verification_failures(samples, mmap)
    
    if keep_passing:
        return samples[~failures]
    else:
        return samples[failures]


def get_verification_success_rate(
    samples: np.ndarray,
    mmap: "MeasurementMap",
) -> float:
    """
    Compute the fraction of shots that passed verification.
    
    Parameters
    ----------
    samples : np.ndarray
        Measurement samples from Stim.
    mmap : MeasurementMap
        Measurement map with verification info.
    
    Returns
    -------
    float
        Success rate in [0, 1].
    """
    if not mmap.verification_measurements or samples.shape[0] == 0:
        return 1.0
    
    failures = extract_verification_failures(samples, mmap)
    return 1.0 - failures.mean()


def create_screening_circuit(
    prep_gadget: "AncillaPrepGadget",
    data_qubits: List[int],
    noise_model: Optional["NoiseModel"] = None,
) -> "stim.Circuit":
    """
    Create a fast screening circuit that only does ancilla prep + verification.
    
    Useful for pre-filtering shots before running the full EC circuit.
    
    Parameters
    ----------
    prep_gadget : AncillaPrepGadget
        The ancilla prep gadget (should be VerifiedAncillaGadget).
    data_qubits : List[int]
        Data qubit indices.
    noise_model : NoiseModel, optional
        Noise model for the prep.
    
    Returns
    -------
    stim.Circuit
        A circuit that prepares ancilla and measures verification, with DETECTORs.
    """
    import stim
    from .ancilla_prep_gadget import VerifiedAncillaGadget, AncillaBasis
    
    if not isinstance(prep_gadget, VerifiedAncillaGadget):
        raise ValueError("Screening circuits require VerifiedAncillaGadget")
    
    circuit = stim.Circuit()
    
    # Prepare and verify
    prep_result = prep_gadget.emit_prepare(
        circuit=circuit,
        ancilla_qubits=data_qubits,
        basis=AncillaBasis.ZERO,
        noise_model=noise_model,
        measurement_offset=0,
        post_select=True,  # Always add DETECTORs for screening
    )
    
    return circuit


# Example usage documentation
__doc__ += """

Example Usage
-------------

>>> from qectostim.experiments.gadgets import TransversalSyndromeGadget, VerifiedAncillaGadget
>>> from qectostim.experiments.gadgets.post_selection_utils import filter_shots
>>> 
>>> # Set up gadget with verified ancillas
>>> gadget = TransversalSyndromeGadget(code=steane_code, ancilla_prep="verified")
>>> mmap = gadget.emit(circuit, data_qubits, noise_model=noise)
>>> 
>>> # Sample
>>> sampler = circuit.compile_sampler()
>>> samples = sampler.sample(10000)
>>> 
>>> # Filter out shots with failed verification
>>> passing_samples = filter_shots(samples, mmap, keep_passing=True)
>>> print(f"Kept {len(passing_samples)}/{len(samples)} shots")
>>> 
>>> # Check success rate
>>> success_rate = get_verification_success_rate(samples, mmap)
>>> print(f"Verification success rate: {success_rate:.2%}")
"""
