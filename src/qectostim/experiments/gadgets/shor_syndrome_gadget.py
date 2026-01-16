"""
Shor-Style Syndrome Extraction Gadget.

This gadget implements Shor's fault-tolerant syndrome extraction protocol,
which uses cat states (GHZ states) and multiple redundant measurements to
achieve robustness against measurement errors.

**Key Difference from Transversal Syndrome Extraction**:
- Transversal: Measures all n ancilla qubits once, then computes syndrome
- Shor: Uses w ancilla qubits per stabilizer (where w = stabilizer weight),
        prepares cat state, extracts parity, and measures each cat qubit
        for redundancy.

**Advantages**:
- Robust to single measurement errors (majority voting)
- Each data qubit interacts with exactly one ancilla per stabilizer
- Single ancilla errors don't spread to multiple data qubits

**Disadvantages**:
- Higher ancilla overhead (need O(w * m) ancillas for m stabilizers)
- More complex circuit with cat state preparation

**Protocol (per stabilizer)**:
1. Prepare w ancilla qubits in |0⟩^⊗w
2. Create cat state: H on first, then CNOT cascade → (|00...0⟩ + |11...1⟩)/√2
3. Apply controlled-Pauli from each ancilla to corresponding data qubit
4. Uncompute cat: reverse CNOT cascade, H on first
5. Measure all w ancillas → should all agree if no errors
6. Majority vote for fault-tolerance

References:
- Shor, "Fault-tolerant quantum computation" (1996)
- Aliferis, Gottesman, Preskill, "Quantum accuracy threshold for
  concatenated distance-3 codes" (2006)
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
import warnings

import stim
import numpy as np

from .base import Gadget, MeasurementMap, SyndromeSchedule, LogicalMeasurementMap
from .noop_gadget import _get_z_support

if TYPE_CHECKING:
    from qectostim.codes.abstract_css import CSSCode
    from qectostim.noise.models import NoiseModel


@dataclass
class CatStateConfig:
    """Configuration for cat state preparation."""
    # Whether to verify cat state before use (adds overhead)
    verify_cat: bool = False
    # Number of repetitions for redundant measurement
    measurement_reps: int = 1


class ShorSyndromeGadget(Gadget):
    """
    Shor-style fault-tolerant syndrome extraction gadget.
    
    This implements Shor's protocol for fault-tolerant syndrome extraction
    using cat states (GHZ states) with multiple redundant measurements.
    
    For each stabilizer S with support on qubits {q1, q2, ..., qw}:
    
    1. **Cat state preparation**:
       - Prepare w ancillas in |0⟩^⊗w
       - Apply H to first ancilla
       - Apply CNOT cascade: CNOT(a_1 → a_2), CNOT(a_1 → a_3), ...
       - Result: (|00...0⟩ + |11...1⟩)/√2
       
    2. **Parity extraction**:
       - For X stabilizer: CNOT(a_i → data_qi) for each qubit in support
       - For Z stabilizer: CNOT(data_qi → a_i) for each qubit in support
       
    3. **Cat state uncomputation**:
       - Reverse CNOT cascade
       - Apply H to first ancilla
       
    4. **Measurement**:
       - Measure all w ancillas in Z basis
       - All should agree (all 0 or all 1) if no errors
       
    5. **Majority voting**:
       - Take majority of w measurements as syndrome bit
       - Robust to single measurement errors
    
    Parameters
    ----------
    code : CSSCode, optional
        The CSS code whose stabilizers to measure.
    extract_x_syndrome : bool
        Whether to extract X stabilizer syndromes (detects Z errors).
    extract_z_syndrome : bool
        Whether to extract Z stabilizer syndromes (detects X errors).
    verify_cat : bool
        Whether to verify cat states before use (adds overhead but more FT).
    measurement_reps : int
        Number of times to repeat the entire syndrome extraction.
        Each repetition uses fresh cat states.
    """
    
    def __init__(
        self,
        code: Optional[Any] = None,
        extract_x_syndrome: bool = True,
        extract_z_syndrome: bool = True,
        verify_cat: bool = False,
        measurement_reps: int = 1,
        ancilla_prep: Optional[str] = None,
    ):
        self.code = code
        self.extract_x_syndrome = extract_x_syndrome
        self.extract_z_syndrome = extract_z_syndrome
        self.verify_cat = verify_cat
        self.measurement_reps = max(1, measurement_reps)
        # Ancilla preparation method: "bare" (default), "encoded", or "verified"
        self.ancilla_prep = ancilla_prep or "bare"
        
        # Create ancilla prep gadget if needed for encoded/verified
        self._ancilla_prep_gadget = None
        if self.ancilla_prep in ("encoded", "verified"):
            from .ancilla_prep_gadget import create_ancilla_prep_gadget
            self._ancilla_prep_gadget = create_ancilla_prep_gadget(self.ancilla_prep, code=code)
        
        # Get parity check matrices if code is provided
        self._hz = None
        self._hx = None
        if code is not None:
            if hasattr(code, 'hz'):
                hz = code.hz
                self._hz = np.atleast_2d(np.array(hz() if callable(hz) else hz, dtype=int))
            if hasattr(code, 'hx'):
                hx = code.hx
                self._hx = np.atleast_2d(np.array(hx() if callable(hx) else hx, dtype=int))
    
    @property
    def name(self) -> str:
        suffix = f"_verified" if self.verify_cat else ""
        rep_suffix = f"_x{self.measurement_reps}" if self.measurement_reps > 1 else ""
        return f"ShorSyndrome{suffix}{rep_suffix}"
    
    @property
    def requires_ancillas(self) -> bool:
        return True
    
    @property
    def ancillas_per_block(self) -> int:
        """
        Calculate ancillas needed per block.
        
        For Shor EC, we need sum of stabilizer weights for both X and Z,
        multiplied by measurement_reps.
        """
        if self._hz is None and self._hx is None:
            return 0
        
        total = 0
        if self.extract_z_syndrome and self._hz is not None:
            # Each Z stabilizer needs weight(stabilizer) ancillas
            for row_idx in range(self._hz.shape[0]):
                total += int(np.sum(self._hz[row_idx, :]))
        if self.extract_x_syndrome and self._hx is not None:
            # Each X stabilizer needs weight(stabilizer) ancillas
            for row_idx in range(self._hx.shape[0]):
                total += int(np.sum(self._hx[row_idx, :]))
        
        return total * self.measurement_reps
    
    @property
    def changes_data_identity(self) -> bool:
        """Data qubits are preserved."""
        return False
    
    def _emit_cat_state_prep(
        self,
        circuit: stim.Circuit,
        ancilla_block_qubits: List[int],
        cat_size: int,
        noise_model: Optional["NoiseModel"] = None,
    ) -> None:
        """
        Prepare cat state (|00...0⟩ + |11...1⟩)/√2 on first cat_size ancilla qubits.
        
        For encoded ancillas: ancilla_block_qubits must be a full code block (n=7 for Steane),
        and we prepare the entire block, then use only first cat_size qubits for cat state.
        
        For bare ancillas: ancilla_block_qubits can be just cat_size qubits.
        
        Parameters
        ----------
        ancilla_block_qubits : List[int]
            Full ancilla block (n=7 for encoded) or cat-sized list (w for bare)
        cat_size : int
            Number of qubits to use for cat state (usually stabilizer weight)
        
        Circuit:
        - Reset all ancillas in block to |0⟩ (using encoded prep if configured)
        - H on first ancilla of cat
        - CNOT from first to each other ancilla in cat
        """
        if cat_size == 0:
            return
        
        # For encoded prep, we need full block size (n=7 for Steane)
        if self.ancilla_prep in ("encoded", "verified") and self._ancilla_prep_gadget is not None:
            # Verify we have a full block
            code_n = self.code.n if self.code is not None else 7
            if len(ancilla_block_qubits) != code_n:
                raise ValueError(
                    f"Encoded ancilla prep requires full code block ({code_n} qubits), "
                    f"got {len(ancilla_block_qubits)} qubits"
                )
            
            # Import AncillaBasis enum
            from .ancilla_prep_gadget import AncillaBasis
            # Use encoded ancilla preparation for the entire block
            self._ancilla_prep_gadget.emit_prepare(
                circuit=circuit,
                ancilla_qubits=ancilla_block_qubits,  # Prepare full block
                basis=AncillaBasis.ZERO,  # Cat state starts in Z basis (|0⟩)
                code=self.code,
                noise_model=noise_model,
            )
            # Extract first cat_size qubits for cat state operations
            cat_qubits = ancilla_block_qubits[:cat_size]
        else:
            # Bare reset (not fault-tolerant)
            # For bare, ancilla_block_qubits should already be cat_size length
            cat_qubits = ancilla_block_qubits
            circuit.append("R", cat_qubits)
            if noise_model is not None and hasattr(noise_model, 'p1') and noise_model.p1 > 0:
                circuit.append("DEPOLARIZE1", cat_qubits, noise_model.p1)
        
        # H on first ancilla
        circuit.append("H", [cat_qubits[0]])
        if noise_model is not None and hasattr(noise_model, 'p1') and noise_model.p1 > 0:
            circuit.append("DEPOLARIZE1", [cat_qubits[0]], noise_model.p1)
        
        # CNOT cascade from first to all others
        for i in range(1, cat_size):
            circuit.append("CX", [cat_qubits[0], cat_qubits[i]])
            if noise_model is not None and hasattr(noise_model, 'p2') and noise_model.p2 > 0:
                circuit.append("DEPOLARIZE2", [cat_qubits[0], cat_qubits[i]], noise_model.p2)
    
    def _emit_cat_state_uncompute(
        self,
        circuit: stim.Circuit,
        ancilla_block_qubits: List[int],
        cat_size: int,
        noise_model: Optional["NoiseModel"] = None,
    ) -> None:
        """
        Uncompute cat state: reverse of preparation.
        
        Only operates on first cat_size qubits from the ancilla block.
        
        Circuit:
        - Reverse CNOT cascade
        - H on first ancilla
        """
        if cat_size == 0:
            return
        
        # Extract cat qubits (first cat_size from block)
        cat_qubits = ancilla_block_qubits[:cat_size]
        
        # Reverse CNOT cascade
        for i in range(cat_size - 1, 0, -1):
            circuit.append("CX", [cat_qubits[0], cat_qubits[i]])
            if noise_model is not None and hasattr(noise_model, 'p2') and noise_model.p2 > 0:
                circuit.append("DEPOLARIZE2", [cat_qubits[0], cat_qubits[i]], noise_model.p2)
        
        # H on first ancilla
        circuit.append("H", [cat_qubits[0]])
        if noise_model is not None and hasattr(noise_model, 'p1') and noise_model.p1 > 0:
            circuit.append("DEPOLARIZE1", [cat_qubits[0]], noise_model.p1)
    
    def _emit_z_syndrome_extraction(
        self,
        circuit: stim.Circuit,
        data_qubits: List[int],
        ancilla_start: int,
        noise_model: Optional["NoiseModel"] = None,
        measurement_offset: int = 0,
    ) -> Tuple[int, List[List[int]]]:
        """
        Extract Z syndrome (detects X errors) using Shor's cat state method.
        
        For each Z stabilizer row in Hz:
        1. Get support qubits
        2. Prepare cat state on w ancillas (w = stabilizer weight)
        3. CNOT from each data qubit in support to corresponding ancilla
        4. Uncompute cat state
        5. Measure all ancillas (majority vote gives syndrome bit)
        
        Returns:
            (next_ancilla_start, measurement_indices_per_stabilizer)
        """
        if self._hz is None:
            return ancilla_start, []
        
        current_ancilla = ancilla_start
        current_meas = measurement_offset
        all_meas_indices = []  # List of lists: [stab_idx][rep_idx][ancilla_within_rep]
        
        for stab_idx in range(self._hz.shape[0]):
            # Get support of this Z stabilizer
            row = self._hz[stab_idx, :]
            support_indices = list(np.where(row)[0])
            support_data = [data_qubits[i] for i in support_indices if i < len(data_qubits)]
            w = len(support_data)
            
            if w == 0:
                all_meas_indices.append([])
                continue
            
            stab_meas = []  # All measurement indices for this stabilizer
            
            for rep in range(self.measurement_reps):
                # Allocate ancillas for this stabilizer
                # For encoded prep: allocate full code block (n=7 for Steane)
                # For bare prep: allocate exactly w ancillas
                if self.ancilla_prep in ("encoded", "verified") and self.code is not None:
                    code_n = self.code.n
                    ancilla_block = list(range(current_ancilla, current_ancilla + code_n))
                    current_ancilla += code_n
                else:
                    ancilla_block = list(range(current_ancilla, current_ancilla + w))
                    current_ancilla += w
                
                # Step 1: Prepare cat state (on full block for encoded, or w qubits for bare)
                self._emit_cat_state_prep(circuit, ancilla_block, w, noise_model)
                circuit.append("TICK")
                
                # Step 2: CNOT from each data qubit to corresponding ancilla
                # Use first w qubits from ancilla_block for coupling
                cat_qubits = ancilla_block[:w]
                for i, (data_q, anc_q) in enumerate(zip(support_data, cat_qubits)):
                    circuit.append("CX", [data_q, anc_q])
                    if noise_model is not None and hasattr(noise_model, 'p2') and noise_model.p2 > 0:
                        circuit.append("DEPOLARIZE2", [data_q, anc_q], noise_model.p2)
                circuit.append("TICK")
                
                # Step 3: Uncompute cat state
                self._emit_cat_state_uncompute(circuit, ancilla_block, w, noise_model)
                circuit.append("TICK")
                
                # Step 4: Measure ancillas used in cat state
                if noise_model is not None and hasattr(noise_model, 'before_measure_flip') and noise_model.before_measure_flip > 0:
                    circuit.append("X_ERROR", cat_qubits, noise_model.before_measure_flip)
                circuit.append("M", cat_qubits)
                
                # Record measurement indices
                rep_meas = list(range(current_meas, current_meas + w))
                stab_meas.extend(rep_meas)
                current_meas += w
                
                circuit.append("TICK")
            
            all_meas_indices.append(stab_meas)
        
        return current_ancilla, all_meas_indices
    
    def _emit_x_syndrome_extraction(
        self,
        circuit: stim.Circuit,
        data_qubits: List[int],
        ancilla_start: int,
        noise_model: Optional["NoiseModel"] = None,
        measurement_offset: int = 0,
    ) -> Tuple[int, List[List[int]]]:
        """
        Extract X syndrome (detects Z errors) using Shor's cat state method.
        
        For each X stabilizer row in Hx:
        1. Get support qubits
        2. Prepare cat state on w ancillas: (|00...0⟩ + |11...1⟩)/√2
        3. CNOT from each ancilla to corresponding data qubit (ancilla controls!)
        4. Uncompute cat state
        5. Measure first ancilla only (gives syndrome bit)
        
        Key insight: For X syndrome, the cat state extracts X parity via 
        CNOT(ancilla→data). The syndrome appears only on the FIRST ancilla
        after uncomputation - other ancillas are always 0 for correct operation.
        
        Note: This modifies the data by X⊗X⊗...⊗X if syndrome is 1, but
        this is a stabilizer operation so preserves the code space.
        
        Returns:
            (next_ancilla_start, measurement_indices_per_stabilizer)
        """
        if self._hx is None:
            return ancilla_start, []
        
        current_ancilla = ancilla_start
        current_meas = measurement_offset
        all_meas_indices = []
        
        for stab_idx in range(self._hx.shape[0]):
            # Get support of this X stabilizer
            row = self._hx[stab_idx, :]
            support_indices = list(np.where(row)[0])
            support_data = [data_qubits[i] for i in support_indices if i < len(data_qubits)]
            w = len(support_data)
            
            if w == 0:
                all_meas_indices.append([])
                continue
            
            stab_meas = []
            
            for rep in range(self.measurement_reps):
                # Allocate ancillas for this stabilizer
                # For encoded prep: allocate full code block (n=7 for Steane)
                # For bare prep: allocate exactly w ancillas
                if self.ancilla_prep in ("encoded", "verified") and self.code is not None:
                    code_n = self.code.n
                    ancilla_block = list(range(current_ancilla, current_ancilla + code_n))
                    current_ancilla += code_n
                else:
                    ancilla_block = list(range(current_ancilla, current_ancilla + w))
                    current_ancilla += w
                
                # Step 1: Prepare cat state (|00...0⟩ + |11...1⟩)/√2
                self._emit_cat_state_prep(circuit, ancilla_block, w, noise_model)
                circuit.append("TICK")
                
                # Step 2: CNOT from each ancilla to corresponding data qubit
                # ANCILLA IS CONTROL - this extracts X parity
                # Use first w qubits from ancilla_block for coupling
                cat_qubits = ancilla_block[:w]
                for i, (data_q, anc_q) in enumerate(zip(support_data, cat_qubits)):
                    circuit.append("CX", [anc_q, data_q])  # ancilla controls data!
                    if noise_model is not None and hasattr(noise_model, 'p2') and noise_model.p2 > 0:
                        circuit.append("DEPOLARIZE2", [anc_q, data_q], noise_model.p2)
                circuit.append("TICK")
                
                # Step 3: Uncompute cat state
                self._emit_cat_state_uncompute(circuit, ancilla_block, w, noise_model)
                circuit.append("TICK")
                
                # Step 4: Measure ancillas used in cat state
                # For X syndrome: first ancilla = syndrome bit, others should be 0
                if noise_model is not None and hasattr(noise_model, 'before_measure_flip') and noise_model.before_measure_flip > 0:
                    circuit.append("X_ERROR", cat_qubits, noise_model.before_measure_flip)
                circuit.append("M", cat_qubits)
                
                # Record measurement indices (first one is the syndrome bit)
                rep_meas = list(range(current_meas, current_meas + w))
                stab_meas.extend(rep_meas)
                current_meas += w
                
                circuit.append("TICK")
            
            all_meas_indices.append(stab_meas)
        
        return current_ancilla, all_meas_indices
    
    def emit(
        self,
        circuit: stim.Circuit,
        data_qubits: Dict[int, List[int]],
        ancilla_qubits: Optional[Dict[int, List[int]]] = None,
        noise_model: Optional["NoiseModel"] = None,
        measurement_offset: int = 0,
    ) -> MeasurementMap:
        """
        Emit Shor-style syndrome extraction for each block.
        
        This uses cat states with multiple ancilla measurements per stabilizer,
        providing robustness against measurement errors through majority voting.
        
        The decoder should take majority vote of the redundant measurements
        to determine each syndrome bit.
        """
        mmap = MeasurementMap(offset=measurement_offset)
        mmap.stabilizer_measurements = {'X': {}, 'Z': {}}
        mmap.measurement_type = "shor_redundant"  # Special flag for decoder
        mmap.shor_measurement_info = {}  # Info for decoder to reconstruct syndromes
        current_meas_idx = measurement_offset
        
        # Allocate ancillas after all data qubits
        max_data_q = max(max(qs) for qs in data_qubits.values()) if data_qubits else -1
        ancilla_start = max_data_q + 1
        
        for block_id in sorted(data_qubits.keys()):
            data_qs = data_qubits[block_id]
            
            # === Z Syndrome Extraction (detects X errors) ===
            if self.extract_z_syndrome:
                ancilla_start, z_meas_lists = self._emit_z_syndrome_extraction(
                    circuit, data_qs, ancilla_start, noise_model, current_meas_idx
                )
                # Flatten to get all Z measurement indices for this block
                z_meas_flat = []
                for stab_meas in z_meas_lists:
                    z_meas_flat.extend(stab_meas)
                mmap.stabilizer_measurements['Z'][block_id] = z_meas_flat
                
                # Store structure info for decoder
                mmap.shor_measurement_info[('Z', block_id)] = {
                    'measurement_lists': z_meas_lists,  # Per-stabilizer lists
                    'measurement_reps': self.measurement_reps,
                    'stabilizer_weights': [len(m) // self.measurement_reps for m in z_meas_lists],
                }
                current_meas_idx += sum(len(m) for m in z_meas_lists)
            
            # === X Syndrome Extraction (detects Z errors) ===
            if self.extract_x_syndrome:
                ancilla_start, x_meas_lists = self._emit_x_syndrome_extraction(
                    circuit, data_qs, ancilla_start, noise_model, current_meas_idx
                )
                # Flatten to get all X measurement indices for this block
                x_meas_flat = []
                for stab_meas in x_meas_lists:
                    x_meas_flat.extend(stab_meas)
                mmap.stabilizer_measurements['X'][block_id] = x_meas_flat
                
                # Store structure info for decoder
                mmap.shor_measurement_info[('X', block_id)] = {
                    'measurement_lists': x_meas_lists,
                    'measurement_reps': self.measurement_reps,
                    'stabilizer_weights': [len(m) // self.measurement_reps for m in x_meas_lists],
                }
                current_meas_idx += sum(len(m) for m in x_meas_lists)
        
        mmap.total_measurements = current_meas_idx - measurement_offset
        return mmap
    
    def get_syndrome_schedule(self) -> SyndromeSchedule:
        """Returns the syndrome extraction schedule."""
        types = []
        rounds = {}
        schedule = []
        
        if self.extract_z_syndrome:
            types.append('Z')
            rounds['Z'] = 1
            schedule.append(('Z', 0))
        if self.extract_x_syndrome:
            types.append('X')
            rounds['X'] = 1
            schedule.append(('X', 0))
        
        return SyndromeSchedule(
            stabilizer_types=types,
            rounds_per_type=rounds,
            schedule=schedule,
        )
    
    def get_syndrome_to_stabilizer_map(
        self,
        syndrome_measurements: List[int],
        stabilizer_type: str,
        code: "CSSCode",
    ) -> Dict[int, List[int]]:
        """
        Map Shor EC syndrome measurements to stabilizer indices.
        
        For Shor EC, each stabilizer has `weight * measurement_reps` measurements.
        These measurements should all agree (form a cat state measurement).
        
        Parameters
        ----------
        syndrome_measurements : List[int]
            Flat list of all measurement indices for this syndrome type
        stabilizer_type : str
            'X' or 'Z' syndrome type
        code : CSSCode
            The code being protected
            
        Returns
        -------
        Dict[int, List[int]]
            Mapping from stabilizer_idx -> list of measurement indices
        """
        # Get parity check matrix
        if stabilizer_type == 'Z':
            H = self._hz if self._hz is not None else (
                np.atleast_2d(np.array(code.hz() if callable(code.hz) else code.hz, dtype=int))
                if hasattr(code, 'hz') else None
            )
        else:
            H = self._hx if self._hx is not None else (
                np.atleast_2d(np.array(code.hx() if callable(code.hx) else code.hx, dtype=int))
                if hasattr(code, 'hx') else None
            )
        
        if H is None:
            # Fallback to one-to-one
            return {i: [m] for i, m in enumerate(syndrome_measurements)}
        
        result = {}
        idx = 0
        for stab_idx in range(H.shape[0]):
            # Weight of this stabilizer
            weight = int(np.sum(H[stab_idx, :]))
            # Total measurements for this stabilizer
            n_meas = weight * self.measurement_reps
            
            if idx + n_meas <= len(syndrome_measurements):
                result[stab_idx] = syndrome_measurements[idx:idx + n_meas]
            else:
                # Fallback: take remaining measurements
                result[stab_idx] = syndrome_measurements[idx:]
            
            idx += n_meas
        
        return result
        
        return SyndromeSchedule(
            stabilizer_types=types,
            rounds_per_type=rounds,
            schedule=schedule,
        )
    
    def get_logical_map(
        self,
        inner_code: "CSSCode",
        outer_code: "CSSCode",
    ) -> LogicalMeasurementMap:
        """Get logical measurement mapping."""
        lmap = LogicalMeasurementMap()
        
        inner_z_support = _get_z_support(inner_code)
        for block_id in range(outer_code.n):
            lmap.inner_z_support[block_id] = inner_z_support
        
        lmap.outer_z_support = _get_z_support(outer_code)
        return lmap


def decode_shor_syndrome(
    measurements: np.ndarray,
    measurement_info: Dict,
    syndrome_type: str = 'Z',
) -> Tuple[int, ...]:
    """
    Decode Shor-style redundant measurements to get syndrome bits.
    
    This function takes the raw measurements from Shor syndrome extraction
    and extracts the actual syndrome.
    
    Parameters
    ----------
    measurements : np.ndarray
        Raw measurement outcomes for one block
    measurement_info : Dict
        Structure info from ShorSyndromeGadget.emit()
    syndrome_type : str
        Either 'Z' (for Z syndrome, uses majority vote) or 'X' (for X syndrome,
        uses first ancilla only).
        
    Returns
    -------
    syndrome : Tuple[int, ...]
        The decoded syndrome bits (one per stabilizer)
        
    Notes
    -----
    For Z syndrome (detects X errors):
        - All cat state ancillas should give same result
        - Use majority voting for fault-tolerance
        
    For X syndrome (detects Z errors):
        - Only first ancilla gives syndrome bit (others should be 0)
        - The CNOT(ancilla→data) pattern means X parity appears on first ancilla only
    """
    measurement_lists = measurement_info['measurement_lists']
    measurement_reps = measurement_info.get('measurement_reps', 1)
    weight = measurement_info.get('weight', None)  # ancillas per stabilizer
    
    syndrome = []
    for stab_idx, stab_meas_indices in enumerate(measurement_lists):
        if not stab_meas_indices:
            syndrome.append(0)
            continue
        
        # Get measurements for this stabilizer
        stab_meas = [int(measurements[i]) for i in stab_meas_indices if i < len(measurements)]
        
        if not stab_meas:
            syndrome.append(0)
            continue
        
        if syndrome_type == 'X':
            # For X syndrome: first ancilla of each rep gives syndrome
            # Other ancillas should be 0 (error detection)
            if weight is not None:
                # Take first ancilla from each repetition
                first_ancillas = stab_meas[::weight]  # every w-th measurement
                syndrome_bit = 1 if sum(first_ancillas) > len(first_ancillas) // 2 else 0
            else:
                # Fallback: just use first measurement
                syndrome_bit = stab_meas[0]
        else:
            # For Z syndrome: all ancillas in cat state should give same result
            # Take majority vote across all measurements.
            syndrome_bit = 1 if sum(stab_meas) > len(stab_meas) // 2 else 0
        
        syndrome.append(syndrome_bit)
    
    return tuple(syndrome)
