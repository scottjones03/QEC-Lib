"""
Teleportation EC Gadget: Teleportation-based Error Correction.

This implements Knill's teleportation-based EC protocol as described by Gottesman:
"An Introduction to Quantum Error Correction and Fault-Tolerant Quantum Computation"

Protocol:
=========
1. Prepare encoded Bell pair on two ancilla blocks:
   |Φ+⟩_L = (|0_L 0_L⟩ + |1_L 1_L⟩)/√2
   
2. Bell measurement between DATA and first ancilla block:
   - CNOT(data → ancilla1)  -- transversal CNOT
   - H(data)                -- transversal Hadamard  
   - Measure data in Z basis → m1 (n-bit classical outcome)
   - Measure ancilla1 in Z basis → m2 (n-bit classical outcome)
   
3. The second ancilla block now contains the teleported data:
   |ψ'⟩ = X_L^f(m1) Z_L^g(m2) |ψ_original⟩
   where f(m1), g(m2) are determined by the logical parities

Key Properties:
===============
- The original data qubits are DESTROYED (measured)
- The output data lives on ancilla2 qubits
- Any errors on original data are detected via syndrome
- The teleportation "filters" errors: ancilla2 starts fresh

For Memory Experiments:
=======================
After each teleportation EC round, the "data" changes identity:
- Round 0: data = qubits [0..n-1]
- Round 1: data = qubits [n..2n-1] (ancilla2 from round 0)

This is tracked via the MeasurementMap.output_qubits field.

CRITICAL: Pauli Frame Tracking
==============================
The teleported state has a Pauli frame:
    |ψ_output⟩ = X^{m_anc1} Z^{m_data} |ψ_original⟩

For Z_L measurement on the output, the decoder MUST account for X frame:
    corrected_output[i] = output_measurement[i] XOR m_anc1[i]
    
Then compute Z_L on the corrected output.

The MeasurementMap provides:
- stabilizer_measurements['Z'][block_id]: indices of m_anc1 (X Pauli frame)
- stabilizer_measurements['X'][block_id]: indices of m_data (Z Pauli frame)
- pauli_frame['X'][block_id]: same as stabilizer_measurements['Z'] for convenience
- output_qubits[block_id]: qubit indices for output measurements

Example decoder logic:
    for i in z_l_support:
        z_l_parity ^= output[i] ^ m_anc1[i]  # Include Pauli frame
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, Optional, Any, Union
import warnings

import stim
import numpy as np

from .base import Gadget, MeasurementMap, SyndromeSchedule, LogicalMeasurementMap
from .noop_gadget import _get_z_support
from .ancilla_prep_gadget import (
    AncillaPrepGadget, AncillaPrepMethod, AncillaBasis,
    BareAncillaGadget, EncodedAncillaGadget, VerifiedAncillaGadget,
    create_ancilla_prep_gadget,
)

if TYPE_CHECKING:
    from qectostim.codes.abstract_css import CSSCode
    from qectostim.noise.models import NoiseModel
    from .combinators import GateImplementationStrategy


class TeleportationECGadget(Gadget):
    """
    Teleportation-based EC gadget (Knill/Gottesman protocol).
    
    This implements the teleportation-based error correction where:
    - The data is teleported through an encoded Bell pair
    - The original data qubits are measured and destroyed
    - The output lives on fresh ancilla qubits
    
    The MeasurementMap tracks:
    - 'X': measurements of original data (Bell measurement X syndrome)
    - 'Z': measurements of ancilla1 (Bell measurement Z syndrome)
    - output_qubits: mapping from block_id to new data qubit locations
    - pauli_frame: classical corrections needed for proper decoding
    - verification_measurements: measurements from ancilla verification (for FT)
    
    Parameters
    ----------
    code : CSSCode
        The CSS code for the inner blocks.
    ancilla_prep : str, AncillaPrepMethod, or AncillaPrepGadget, optional
        How to prepare ancilla qubits for Bell pairs. Options:
        - "bare" (default): Simple reset, NOT fault-tolerant
        - "encoded": Proper encoding circuit, fault-tolerant
        - "verified": Encoding + verification, fully fault-tolerant
    gate_strategy : GateImplementationStrategy, optional
        Strategy for implementing logical gates (default: TransversalStrategy).
    
    Note
    ----
    For fault-tolerant Knill EC, use ancilla_prep="encoded" or "verified".
    The "verified" option adds detectors based on ancilla verification.
    
    Because this gadget destroys the original data and outputs to new qubits,
    experiments using this gadget must track qubit identity across rounds.
    
    The decoder must also account for the Pauli frame - see module docstring.
    """
    
    def __init__(
        self,
        code: Any,
        ancilla_prep: Optional[Union[str, AncillaPrepMethod, AncillaPrepGadget]] = None,
        gate_strategy: Optional["GateImplementationStrategy"] = None,
    ):
        from .combinators import TransversalStrategy
        
        self.code = code
        self.gate_strategy = gate_strategy or TransversalStrategy()
        
        # Set up ancilla preparation gadget
        if ancilla_prep is None:
            # Default: bare (backward compatible, but NOT FT)
            self.ancilla_prep = BareAncillaGadget()
        elif isinstance(ancilla_prep, AncillaPrepGadget):
            self.ancilla_prep = ancilla_prep
        elif isinstance(ancilla_prep, (str, AncillaPrepMethod)):
            self.ancilla_prep = create_ancilla_prep_gadget(ancilla_prep, code=code)
        else:
            raise ValueError(f"Invalid ancilla_prep: {ancilla_prep}")
        
        # Get parity check matrices
        hx = code.hx
        hz = code.hz
        self.hx = np.atleast_2d(np.array(hx() if callable(hx) else hx, dtype=int))
        self.hz = np.atleast_2d(np.array(hz() if callable(hz) else hz, dtype=int))
    
    @property
    def name(self) -> str:
        strategy = self.gate_strategy.name if self.gate_strategy.name != "Transversal" else ""
        prep_suffix = f"_{self.ancilla_prep.name}" if self.ancilla_prep.name != "BareAncilla" else ""
        return f"TeleportationEC{('_' + strategy) if strategy else ''}{prep_suffix}"
    
    @property
    def requires_ancillas(self) -> bool:
        return True
    
    @property
    def ancillas_per_block(self) -> int:
        """Need 2 ancilla blocks per data block for Bell pair."""
        return 2 * self.code.n
    
    @property
    def changes_data_identity(self) -> bool:
        """True because output qubits differ from input qubits."""
        return True
    
    def emit(
        self,
        circuit: stim.Circuit,
        data_qubits: Dict[int, List[int]],
        ancilla_qubits: Optional[Dict[int, List[int]]] = None,
        noise_model: Optional["NoiseModel"] = None,
        measurement_offset: int = 0,
    ) -> MeasurementMap:
        """
        Emit teleportation EC with fault-tolerant ancilla preparation.
        
        For each data block:
        1. Prepare ancilla1 in |0_L⟩ and ancilla2 in |+_L⟩ (using ancilla_prep)
        2. Create Bell pair: CNOT(ancilla2 → ancilla1) gives |Φ+⟩_L
        3. Bell measurement: CNOT(data → ancilla1), H(data), M(data), M(ancilla1)
        4. Output tracking: ancilla2 now holds teleported data
        
        When using "verified" ancilla_prep, verification measurements are added
        as detectors to catch ancilla preparation errors.
        
        Returns
        -------
        MeasurementMap with:
            - stabilizer_measurements['X']: data measurement indices (m1)
            - stabilizer_measurements['Z']: ancilla1 measurement indices (m2)
            - output_qubits: {block_id: [qubit indices of ancilla2]}
            - pauli_frame['X']: same as stabilizer_measurements['Z'] for decoder
            - verification_measurements: {block_id: [verification meas indices]}
        """
        mmap = MeasurementMap(offset=measurement_offset)
        mmap.stabilizer_measurements = {'X': {}, 'Z': {}}
        mmap.output_qubits = {}
        mmap.pauli_frame = {'X': {}, 'Z': {}}  # Track Pauli frame for decoder
        mmap.verification_measurements = {}  # For FT ancilla verification
        current_meas = measurement_offset
        
        max_data_q = max(max(qs) for qs in data_qubits.values()) if data_qubits else -1
        ancilla_start = max_data_q + 1
        
        for block_id, data_qs in sorted(data_qubits.items()):
            n = len(data_qs)
            
            # Allocate two ancilla blocks
            ancilla1 = list(range(ancilla_start, ancilla_start + n))
            ancilla2 = list(range(ancilla_start + n, ancilla_start + 2 * n))
            ancilla_start += 2 * n
            
            block_verification_meas = []
            
            # =========== Step 1: Prepare Bell pair with FT ancilla prep ===========
            # For FT: prepare |0_L⟩ on ancilla1, |+_L⟩ on ancilla2
            # Then CNOT(ancilla2 → ancilla1) creates encoded Bell pair |Φ+⟩_L
            
            # Prepare ancilla1 in |0_L⟩ (or |0⟩^⊗n for bare prep)
            prep_result1 = self.ancilla_prep.emit_prepare(
                circuit=circuit,
                ancilla_qubits=ancilla1,
                basis=AncillaBasis.ZERO,
                code=self.code,
                noise_model=noise_model,
                measurement_offset=current_meas,
            )
            if prep_result1.verification_measurements:
                block_verification_meas.extend(prep_result1.verification_measurements)
            current_meas += prep_result1.total_measurements
            
            # Prepare ancilla2 in |+_L⟩ (or |+⟩^⊗n for bare prep)
            prep_result2 = self.ancilla_prep.emit_prepare(
                circuit=circuit,
                ancilla_qubits=ancilla2,
                basis=AncillaBasis.PLUS,
                code=self.code,
                noise_model=noise_model,
                measurement_offset=current_meas,
            )
            if prep_result2.verification_measurements:
                block_verification_meas.extend(prep_result2.verification_measurements)
            current_meas += prep_result2.total_measurements
            
            circuit.append("TICK")
            
            # CNOT(ancilla2 → ancilla1) creates Bell pairs (transversal)
            for i in range(n):
                circuit.append("CX", [ancilla2[i], ancilla1[i]])
                if noise_model:
                    noise_model.apply_two_qubit_noise(circuit, ancilla2[i], ancilla1[i])
            
            circuit.append("TICK")
            
            # =========== Step 2: Bell measurement (data, ancilla1) ===========
            # CNOT(data → ancilla1)
            self.gate_strategy.emit_logical_cnot(circuit, data_qs, ancilla1, noise_model)
            
            circuit.append("TICK")
            
            # H(data)
            self.gate_strategy.emit_logical_h(circuit, data_qs, noise_model)
            
            circuit.append("TICK")
            
            # Measure data in Z basis → X syndrome for teleported state
            circuit.append("M", data_qs)
            data_meas = list(range(current_meas, current_meas + n))
            mmap.stabilizer_measurements['X'][block_id] = data_meas
            current_meas += n
            
            # Measure ancilla1 in Z basis → Z syndrome for teleported state
            circuit.append("M", ancilla1)
            anc1_meas = list(range(current_meas, current_meas + n))
            mmap.stabilizer_measurements['Z'][block_id] = anc1_meas
            # X Pauli frame: output[i] must be XORed with m_anc1[i] for Z_L
            mmap.pauli_frame['X'][block_id] = anc1_meas
            current_meas += n
            
            circuit.append("TICK")
            
            # =========== Step 3: Track output qubits and verification ===========
            # ancilla2 now holds: X_L^f(m1) Z_L^g(m2) |original_data⟩
            mmap.output_qubits[block_id] = ancilla2
            
            # Store verification measurements for this block (if any)
            if block_verification_meas:
                mmap.verification_measurements[block_id] = block_verification_meas
        
        mmap.total_measurements = current_meas - measurement_offset
        return mmap
    
    def get_syndrome_schedule(self) -> SyndromeSchedule:
        """Bell measurement extracts both X and Z syndrome information."""
        return SyndromeSchedule(
            stabilizer_types=['X', 'Z'],
            rounds_per_type={'X': 1, 'Z': 1},
            schedule=[('X', 0), ('Z', 0)],
        )
    
    def get_logical_map(
        self,
        inner_code: "CSSCode",
        outer_code: "CSSCode",
    ) -> LogicalMeasurementMap:
        """Get logical measurement mapping for hierarchical decoding."""
        lmap = LogicalMeasurementMap()
        
        inner_z_support = _get_z_support(inner_code)
        for block_id in range(outer_code.n):
            lmap.inner_z_support[block_id] = inner_z_support
        
        lmap.outer_z_support = _get_z_support(outer_code)
        return lmap


# =============================================================================
# Backward Compatibility Alias
# =============================================================================

class TrueKnillECGadget(TeleportationECGadget):
    """
    DEPRECATED: Use TeleportationECGadget instead.
    
    This is an alias for backward compatibility.
    """
    
    def __init__(self, *args, **kwargs):
        warnings.warn(
            "TrueKnillECGadget is deprecated. Use TeleportationECGadget instead.",
            DeprecationWarning,
            stacklevel=2
        )
        super().__init__(*args, **kwargs)
    
    @property
    def name(self) -> str:
        return "TrueKnillEC_DEPRECATED"
