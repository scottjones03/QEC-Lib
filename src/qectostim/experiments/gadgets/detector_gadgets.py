"""
Detector-Based Transversal Syndrome Gadget.

This module implements transversal syndrome extraction using the detector-based
architecture. Key differences from the original TransversalSyndromeGadget:

1. **Emits DETECTOR instructions directly** after each measurement block
2. **Returns DetectorMap** instead of MeasurementMap  
3. **Tracks detector indices** via DetectorCounter
4. **No OBSERVABLE_INCLUDE** - logical outcomes computed in Python

The gadget appends:
- Ancilla preparation (bare, encoded, or verified)
- Transversal CNOTs (data ↔ ancilla)
- Measurements of ancilla block
- DETECTOR instructions for syndrome extraction

For verified ancilla prep, adds verification DETECTORs for post-selection.

Usage:
    >>> gadget = DetectorTransversalSyndrome(code)
    >>> counter = DetectorCounter()
    >>> dmap = gadget.emit(circuit, data_qubits, counter)
    >>> # dmap contains detector group ranges for decoding
"""
from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Any, Union

import stim
import numpy as np

from .detector_primitives import (
    DetectorMap, DetectorGroup, DetectorType, DetectorCounter,
    GadgetResult, emit_detector, emit_detector_group, emit_parity_detector,
)
from .ancilla_prep_gadget import (
    AncillaPrepGadget, AncillaPrepMethod, AncillaBasis,
    BareAncillaGadget, EncodedAncillaGadget, VerifiedAncillaGadget,
    create_ancilla_prep_gadget,
)

if TYPE_CHECKING:
    from qectostim.codes.abstract_css import CSSCode
    from qectostim.noise.models import NoiseModel


class DetectorTransversalSyndrome:
    """
    Detector-based transversal syndrome extraction gadget.
    
    This implements Steane-style syndrome extraction where:
    1. Ancilla block prepared in |0_L⟩ or |+_L⟩
    2. Transversal CNOTs transfer syndrome info to ancilla
    3. Ancilla measured, DETECTOR instructions emitted immediately
    
    The gadget emits DETECTORs that can be used directly by the detector sampler.
    No temporal differencing or OBSERVABLE_INCLUDE - just raw detector bits.
    
    Parameters
    ----------
    code : CSSCode, optional
        The CSS code (provides Hz/Hx for syndrome computation)
    extract_x_syndrome : bool
        Whether to extract X stabilizer syndrome (detects Z errors)
    extract_z_syndrome : bool
        Whether to extract Z stabilizer syndrome (detects X errors)
    ancilla_prep : str or AncillaPrepGadget
        Ancilla preparation method:
        - "bare": Simple reset (NOT fault-tolerant)
        - "encoded": Proper encoding (fault-tolerant)
        - "verified": Encoding + verification (fully FT with post-selection)
    emit_individual_detectors : bool
        If True, emit one DETECTOR per syndrome bit (classical syndrome)
        If False, emit n DETECTORs for n ancilla measurements (raw ancilla mode)
    """
    
    def __init__(
        self,
        code: Optional[Any] = None,
        extract_x_syndrome: bool = True,
        extract_z_syndrome: bool = True,
        ancilla_prep: Optional[Union[str, AncillaPrepMethod, AncillaPrepGadget]] = None,
        emit_individual_detectors: bool = False,
    ):
        self.code = code
        self.extract_x_syndrome = extract_x_syndrome
        self.extract_z_syndrome = extract_z_syndrome
        self.emit_individual_detectors = emit_individual_detectors
        
        # Set up ancilla preparation gadget
        if ancilla_prep is None:
            self.ancilla_prep = EncodedAncillaGadget(code=code)
        elif isinstance(ancilla_prep, AncillaPrepGadget):
            self.ancilla_prep = ancilla_prep
        elif isinstance(ancilla_prep, (str, AncillaPrepMethod)):
            self.ancilla_prep = create_ancilla_prep_gadget(ancilla_prep, code=code)
        else:
            raise ValueError(f"Invalid ancilla_prep: {ancilla_prep}")
        
        if isinstance(self.ancilla_prep, BareAncillaGadget):
            warnings.warn(
                "DetectorTransversalSyndrome using bare ancillas; NOT fault-tolerant. "
                "Use ancilla_prep='encoded' or 'verified' for FT.",
                RuntimeWarning,
            )
        
        # Get parity check matrices
        self._hz = None
        self._hx = None
        if code is not None:
            if hasattr(code, 'hz'):
                self._hz = np.array(code.hz, dtype=np.uint8)
            if hasattr(code, 'hx'):
                self._hx = np.array(code.hx, dtype=np.uint8)
    
    @property
    def name(self) -> str:
        prep_suffix = f"_{self.ancilla_prep.name}" if hasattr(self.ancilla_prep, 'name') else ""
        return f"DetectorTransversalSyndrome{prep_suffix}"
    
    @property
    def n_qubits_per_block(self) -> int:
        """Number of data qubits per block."""
        if self.code is not None:
            return self.code.n
        return 7  # Default to Steane
    
    def emit(
        self,
        circuit: stim.Circuit,
        data_qubits: Dict[int, List[int]],
        counter: DetectorCounter,
        ancilla_start: Optional[int] = None,
        noise_model: Optional["NoiseModel"] = None,
        measurement_offset: int = 0,
        round_idx: int = 0,
        level: int = 0,
    ) -> DetectorMap:
        """
        Emit transversal syndrome extraction with DETECTOR instructions.
        
        For each block:
        1. Prepare ancilla in |0_L⟩ (for Z syndrome) or |+_L⟩ (for X syndrome)
        2. Transversal CNOT between data and ancilla
        3. Measure ancilla block
        4. Emit DETECTOR instructions for syndrome bits
        
        Args:
            circuit: Stim circuit to append to
            data_qubits: block_id -> list of data qubit indices
            counter: DetectorCounter to track detector positions
            ancilla_start: First ancilla qubit index (auto-computed if None)
            noise_model: Optional noise model
            measurement_offset: Starting measurement index
            round_idx: EC round index
            level: Concatenation level (for hierarchical codes)
            
        Returns:
            DetectorMap with syndrome and verification detector groups
        """
        dmap = DetectorMap(offset=counter.peek(), measurement_offset=measurement_offset)
        current_meas_idx = measurement_offset
        
        # Allocate ancilla qubits
        if ancilla_start is None:
            max_data_q = max(max(qs) for qs in data_qubits.values()) if data_qubits else -1
            ancilla_start = max_data_q + 1
        
        n_per_block = self.n_qubits_per_block
        
        for block_id in sorted(data_qubits.keys()):
            data_qs = data_qubits[block_id]
            n = len(data_qs)
            
            # Ancilla qubits for this block (reused for Z and X syndrome)
            block_ancilla_start = ancilla_start + block_id * n
            ancilla_qs = list(range(block_ancilla_start, block_ancilla_start + n))
            
            # === Z SYNDROME EXTRACTION (detects X errors) ===
            if self.extract_z_syndrome:
                z_dgroup = self._emit_syndrome_extraction(
                    circuit=circuit,
                    data_qs=data_qs,
                    ancilla_qs=ancilla_qs,
                    basis='Z',
                    block_id=block_id,
                    counter=counter,
                    noise_model=noise_model,
                    current_meas_idx=current_meas_idx,
                    round_idx=round_idx,
                    level=level,
                    dmap=dmap,
                )
                current_meas_idx += n
            
            # === X SYNDROME EXTRACTION (detects Z errors) ===
            if self.extract_x_syndrome:
                x_dgroup = self._emit_syndrome_extraction(
                    circuit=circuit,
                    data_qs=data_qs,
                    ancilla_qs=ancilla_qs,
                    basis='X',
                    block_id=block_id,
                    counter=counter,
                    noise_model=noise_model,
                    current_meas_idx=current_meas_idx,
                    round_idx=round_idx,
                    level=level,
                    dmap=dmap,
                )
                current_meas_idx += n
        
        dmap.total_measurements = current_meas_idx - measurement_offset
        return dmap
    
    def _emit_syndrome_extraction(
        self,
        circuit: stim.Circuit,
        data_qs: List[int],
        ancilla_qs: List[int],
        basis: str,
        block_id: int,
        counter: DetectorCounter,
        noise_model: Optional["NoiseModel"],
        current_meas_idx: int,
        round_idx: int,
        level: int,
        dmap: DetectorMap,
    ) -> DetectorGroup:
        """
        Emit syndrome extraction for one basis (Z or X).
        
        Returns the DetectorGroup for this syndrome extraction.
        """
        n = len(data_qs)
        
        # Step 1: Prepare ancilla
        ancilla_basis = AncillaBasis.ZERO if basis == 'Z' else AncillaBasis.PLUS
        
        prep_result = self.ancilla_prep.emit_prepare(
            circuit=circuit,
            ancilla_qubits=ancilla_qs,
            basis=ancilla_basis,
            code=self.code,
            noise_model=noise_model,
            measurement_offset=current_meas_idx,
        )
        
        # Track verification measurements for post-selection
        if prep_result.verification_measurements:
            ver_start = counter.peek()
            for ver_idx in prep_result.verification_measurements:
                lookback = ver_idx - (current_meas_idx + prep_result.total_measurements)
                circuit.append("DETECTOR", [stim.target_rec(lookback)])
                counter.advance(1)
            ver_end = counter.peek()
            
            ver_group = DetectorGroup(
                start=ver_start,
                end=ver_end,
                dtype=DetectorType.VERIFICATION,
                block_id=block_id,
                level=level,
                round_idx=round_idx,
                metadata={'basis': basis},
            )
            dmap.add_group(ver_group)
        
        current_meas_idx += prep_result.total_measurements
        
        # Step 2: Transversal CNOTs
        # BOTH Z and X syndrome extraction use CNOT(data→ancilla)
        # The difference is ancilla preparation and measurement basis:
        #   Z syndrome: ancilla |0⟩, CNOT(data→ancilla), measure Z → detects X errors
        #   X syndrome: ancilla |+⟩, CNOT(data→ancilla), measure X → detects Z errors
        for d, a in zip(data_qs, ancilla_qs):
            circuit.append("CNOT", [d, a])
        
        # Step 3: Measure ancilla
        if basis == 'Z':
            # Measure in Z basis directly
            circuit.append("M", ancilla_qs)
        else:
            # Measure in X basis: H then M
            circuit.append("H", ancilla_qs)
            circuit.append("M", ancilla_qs)
        
        # Record measurement indices
        meas_indices = list(range(current_meas_idx, current_meas_idx + n))
        
        # Step 4: Emit DETECTOR instructions
        det_start = counter.peek()
        
        if self.emit_individual_detectors and self._get_H(basis) is not None:
            # Emit syndrome-level detectors (one per stabilizer)
            H = self._get_H(basis)
            n_stabilizers = H.shape[0]
            
            for stab_idx in range(n_stabilizers):
                # Find which ancilla qubits contribute to this syndrome bit
                support = np.where(H[stab_idx] == 1)[0]
                contributing_meas = [meas_indices[i] for i in support]
                
                # Emit parity detector
                targets = []
                for idx in contributing_meas:
                    lookback = idx - (current_meas_idx + n)
                    targets.append(stim.target_rec(lookback))
                
                circuit.append("DETECTOR", targets)
                counter.advance(1)
        else:
            # Emit raw ancilla detectors (one per ancilla qubit)
            # Decoder will compute syndrome from these
            for i, meas_idx in enumerate(meas_indices):
                lookback = meas_idx - (current_meas_idx + n)
                circuit.append("DETECTOR", [stim.target_rec(lookback)])
                counter.advance(1)
        
        det_end = counter.peek()
        
        # Create detector group
        dtype = DetectorType.SYNDROME_Z if basis == 'Z' else DetectorType.SYNDROME_X
        dgroup = DetectorGroup(
            start=det_start,
            end=det_end,
            dtype=dtype,
            block_id=block_id,
            level=level,
            round_idx=round_idx,
            metadata={
                'basis': basis,
                'meas_indices': meas_indices,
                'n_ancilla': n,
            },
        )
        dmap.add_group(dgroup)
        
        return dgroup
    
    def _get_H(self, basis: str) -> Optional[np.ndarray]:
        """Get parity check matrix for given basis."""
        if basis == 'Z':
            return self._hz
        else:
            return self._hx


class DetectorVerifiedAncillaPrep:
    """
    Detector-based verified ancilla preparation.
    
    Prepares encoded ancilla with verification measurements,
    emitting DETECTOR instructions for post-selection.
    
    The verification detectors should have outcome 0 for valid preparation.
    Shots where any verification detector fires should be rejected.
    """
    
    def __init__(
        self,
        code: Optional[Any] = None,
        enable_post_selection: bool = True,
    ):
        self.code = code
        self.enable_post_selection = enable_post_selection
        self._encoder = EncodedAncillaGadget(code=code)
        
        # Get stabilizers for verification
        self._x_stabilizers = self._get_stabilizers('X')
        self._z_stabilizers = self._get_stabilizers('Z')
    
    def _get_stabilizers(self, basis: str) -> List[List[int]]:
        """Get stabilizer generator supports for verification."""
        if self.code is None:
            # Default to Steane code stabilizers
            return [
                [0, 2, 4, 6],
                [1, 2, 5, 6],
                [3, 4, 5, 6],
            ]
        
        if basis == 'X':
            if hasattr(self.code, 'hx'):
                return [list(np.where(row)[0]) for row in self.code.hx]
        else:
            if hasattr(self.code, 'hz'):
                return [list(np.where(row)[0]) for row in self.code.hz]
        
        return []
    
    def emit(
        self,
        circuit: stim.Circuit,
        ancilla_qubits: List[int],
        basis: AncillaBasis,
        counter: DetectorCounter,
        noise_model: Optional["NoiseModel"] = None,
        measurement_offset: int = 0,
    ) -> Tuple[DetectorGroup, int]:
        """
        Emit verified ancilla preparation with verification DETECTORs.
        
        For |0_L⟩: encode then measure X stabilizers (should all be +1)
        For |+_L⟩: encode then measure Z stabilizers (should all be +1)
        
        Returns:
            (verification_detector_group, n_measurements)
        """
        n = len(ancilla_qubits)
        current_meas = measurement_offset
        
        # Step 1: Encode ancilla
        prep_result = self._encoder.emit_prepare(
            circuit=circuit,
            ancilla_qubits=ancilla_qubits,
            basis=basis,
            code=self.code,
            noise_model=noise_model,
            measurement_offset=current_meas,
        )
        current_meas += prep_result.total_measurements
        
        # Step 2: Measure verification stabilizers
        # For |0_L⟩: measure X stabilizers (use additional ancilla qubits)
        # For |+_L⟩: measure Z stabilizers
        
        stabilizers = self._x_stabilizers if basis == AncillaBasis.ZERO else self._z_stabilizers
        n_stab = len(stabilizers)
        
        # Allocate verification ancilla qubits
        max_q = max(ancilla_qubits)
        ver_ancilla = list(range(max_q + 1, max_q + 1 + n_stab))
        
        # Prepare verification ancilla in |0⟩
        circuit.append("R", ver_ancilla)
        
        # For X stabilizer measurement: H, CNOT cascade, H, M
        # For Z stabilizer measurement: CNOT cascade, M
        if basis == AncillaBasis.ZERO:
            # Measuring X stabilizers
            circuit.append("H", ver_ancilla)
        
        # CNOT from verification ancilla to data qubits on stabilizer support
        for stab_idx, support in enumerate(stabilizers):
            ver_q = ver_ancilla[stab_idx]
            for data_idx in support:
                target_q = ancilla_qubits[data_idx]
                if basis == AncillaBasis.ZERO:
                    # X measurement: ancilla controls
                    circuit.append("CNOT", [ver_q, target_q])
                else:
                    # Z measurement: data controls
                    circuit.append("CNOT", [target_q, ver_q])
        
        if basis == AncillaBasis.ZERO:
            circuit.append("H", ver_ancilla)
        
        # Measure verification ancilla
        circuit.append("M", ver_ancilla)
        
        ver_meas_indices = list(range(current_meas, current_meas + n_stab))
        current_meas += n_stab
        
        # Step 3: Emit verification DETECTORs
        det_start = counter.peek()
        
        for meas_idx in ver_meas_indices:
            lookback = meas_idx - current_meas
            circuit.append("DETECTOR", [stim.target_rec(lookback)])
            counter.advance(1)
        
        det_end = counter.peek()
        
        ver_group = DetectorGroup(
            start=det_start,
            end=det_end,
            dtype=DetectorType.VERIFICATION,
            block_id=0,
            level=0,
            round_idx=0,
            metadata={'verification_type': 'ancilla_prep', 'basis': basis.value},
        )
        
        n_total_meas = current_meas - measurement_offset
        return ver_group, n_total_meas


class DetectorNoOp:
    """
    No-operation gadget that just applies idle noise.
    
    Useful for memory experiments with no EC rounds.
    Doesn't emit any detectors.
    """
    
    def __init__(self, code: Optional[Any] = None):
        self.code = code
    
    @property
    def name(self) -> str:
        return "DetectorNoOp"
    
    def emit(
        self,
        circuit: stim.Circuit,
        data_qubits: Dict[int, List[int]],
        counter: DetectorCounter,
        noise_model: Optional["NoiseModel"] = None,
        measurement_offset: int = 0,
        round_idx: int = 0,
        level: int = 0,
    ) -> DetectorMap:
        """
        Emit idle noise (no operations, no detectors).
        """
        dmap = DetectorMap(offset=counter.peek(), measurement_offset=measurement_offset)
        
        # Just apply idle noise if provided
        if noise_model is not None:
            all_qubits = []
            for qs in data_qubits.values():
                all_qubits.extend(qs)
            # Apply depolarizing noise as idle
            if hasattr(noise_model, 'idle_error_rate'):
                for q in all_qubits:
                    circuit.append("DEPOLARIZE1", [q], noise_model.idle_error_rate)
        
        return dmap
