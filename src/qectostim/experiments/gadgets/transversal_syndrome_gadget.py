"""
Transversal Syndrome Gadget: Transversal syndrome extraction for CSS codes.

This is the standard Steane-style syndrome extraction protocol:
- Z syndrome: |0⟩ ancilla + CNOT(data→ancilla) + measure Z
- X syndrome: |+⟩ ancilla + CNOT(data→ancilla) + H + measure Z

The key property is that all CNOTs are transversal (qubit-by-qubit parallel),
which prevents error propagation across the code block.

**IMPORTANT**: For true fault-tolerance, the ancilla must be prepared in an
ENCODED state (|0_L⟩ or |+_L⟩), not a bare state (|0⟩^⊗n or |+⟩^⊗n).
Use the `ancilla_prep` parameter to configure ancilla preparation:
- "bare": Simple reset (NOT fault-tolerant, but fast)
- "encoded": Proper encoding (fault-tolerant)
- "verified": Encoding + verification (fully fault-tolerant)

References:
- Steane, "Active Stabilization, Quantum Computation, and Quantum State Synthesis"
- Gottesman, "An Introduction to Quantum Error Correction and Fault-Tolerant 
  Quantum Computation"

Circuit Structure (per block):
┌────────────────────────────────────────────────────────────────────┐
│ Z syndrome:                          X syndrome:                   │
│                                                                    │
│ |0⟩ Ancilla ──R────────CX────M       |+⟩ Ancilla ──R──H──CX──H──M │
│                        │                              │            │
│ Data        ───────────●─────        Data        ─────●────────── │
└────────────────────────────────────────────────────────────────────┘
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Any, Union
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


class TransversalSyndromeGadget(Gadget):
    """
    Transversal syndrome extraction gadget for CSS codes.
    
    This implements the standard Steane-style syndrome extraction protocol
    where syndrome information is extracted using transversal CNOTs with
    an (optionally encoded) ancilla block.
    
    The protocol extracts both X and Z syndromes:
    
    1. **Z syndrome** (detects X errors):
       - Prepare ancilla in |0⟩^n or |0_L⟩ (depending on ancilla_prep)
       - Transversal CNOT(data → ancilla)
       - Measure ancilla in Z basis
       
    2. **X syndrome** (detects Z errors):
       - Prepare ancilla in |+⟩^n or |+_L⟩ (depending on ancilla_prep)
       - Transversal CNOT(data → ancilla)
       - Measure ancilla in X basis (via H then Z measurement)
    
    The ancilla block is reused for both syndrome extractions.
    
    Parameters
    ----------
    code : CSSCode, optional
        The code (provides Hz/Hx for syndrome computation).
    extract_x_syndrome : bool
        Whether to extract X stabilizer syndrome (detects Z errors).
    extract_z_syndrome : bool
        Whether to extract Z stabilizer syndrome (detects X errors).
    ancilla_prep : str, AncillaPrepMethod, or AncillaPrepGadget, optional
        How to prepare ancilla qubits. Options:
        - "bare" (default): Simple reset, NOT fault-tolerant
        - "encoded": Proper encoding circuit, fault-tolerant
        - "verified": Encoding + verification, fully fault-tolerant
        - AncillaPrepGadget instance for custom preparation
    gate_strategy : GateImplementationStrategy, optional
        Strategy for implementing logical gates (default: TransversalStrategy).
    
    Note
    ----
    For fault-tolerant operation, use ancilla_prep="encoded" or "verified".
    The default "bare" preparation is NOT fault-tolerant but is kept for
    backward compatibility and for cases where speed is preferred over FT.
    """
    
    def __init__(
        self,
        code: Optional[Any] = None,
        extract_x_syndrome: bool = True,
        extract_z_syndrome: bool = True,
        ancilla_prep: Optional[Union[str, AncillaPrepMethod, AncillaPrepGadget]] = None,
        gate_strategy: Optional["GateImplementationStrategy"] = None,
    ):
        from .combinators import TransversalStrategy
        
        self.code = code
        self.extract_x_syndrome = extract_x_syndrome
        self.extract_z_syndrome = extract_z_syndrome
        self.gate_strategy = gate_strategy or TransversalStrategy()
        
        # Set up ancilla preparation gadget
        if ancilla_prep is None:
            # Default to encoded ancilla for FT by default
            self.ancilla_prep = EncodedAncillaGadget(code=code)
        elif isinstance(ancilla_prep, AncillaPrepGadget):
            self.ancilla_prep = ancilla_prep
        elif isinstance(ancilla_prep, (str, AncillaPrepMethod)):
            self.ancilla_prep = create_ancilla_prep_gadget(ancilla_prep, code=code)
        else:
            raise ValueError(f"Invalid ancilla_prep: {ancilla_prep}")

        if isinstance(self.ancilla_prep, BareAncillaGadget):
            warnings.warn(
                "TransversalSyndromeGadget using bare ancillas; this is NOT fault-tolerant. "
                "Pass ancilla_prep='encoded' or 'verified' to maintain FT.",
                RuntimeWarning,
            )
        
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
        strategy_suffix = f"_{self.gate_strategy.name}" if self.gate_strategy.name != "Transversal" else ""
        prep_suffix = f"_{self.ancilla_prep.name}" if self.ancilla_prep.name != "BareAncilla" else ""
        return f"TransversalSyndrome{strategy_suffix}{prep_suffix}"
    
    @property
    def requires_ancillas(self) -> bool:
        return True
    
    @property
    def ancillas_per_block(self) -> int:
        """Need n ancillas per block (reused for Z and X syndrome extraction)."""
        if self.code is not None:
            return self.code.n
        return 0
    
    @property
    def changes_data_identity(self) -> bool:
        """Data qubits are preserved."""
        return False
    
    def emit(
        self,
        circuit: stim.Circuit,
        data_qubits: Dict[int, List[int]],
        ancilla_qubits: Optional[Dict[int, List[int]]] = None,
        noise_model: Optional["NoiseModel"] = None,
        measurement_offset: int = 0,
        post_select: Optional[bool] = None,
        strict_verify: bool = True,
    ) -> MeasurementMap:
        """
        Emit transversal syndrome extraction for each block.
        
        For each block:
        1. Allocate n ancillas  
        2. Z syndrome: prepare ancilla in |0_L⟩ + CNOT(data→ancilla) + measure Z
        3. X syndrome: prepare ancilla in |+_L⟩ + CNOT(data→ancilla) + measure X (via H+Z)
        
        The ancilla preparation is controlled by self.ancilla_prep:
        - BareAncillaGadget: Just reset (NOT fault-tolerant)
        - EncodedAncillaGadget: Proper encoding circuit (fault-tolerant)
        - VerifiedAncillaGadget: Encoding + verification (fully fault-tolerant)
        """
        # Default post-select on if using verified ancillas to reduce footguns
        effective_post_select = post_select
        if effective_post_select is None:
            effective_post_select = isinstance(self.ancilla_prep, VerifiedAncillaGadget)
        if strict_verify and isinstance(self.ancilla_prep, VerifiedAncillaGadget) and not effective_post_select:
            raise ValueError("Verified ancillas require post_select=True when strict_verify is enabled.")

        mmap = MeasurementMap(offset=measurement_offset)
        mmap.stabilizer_measurements = {'X': {}, 'Z': {}}
        mmap.verification_measurements = {}
        mmap.measurement_type = "raw_ancilla"  # We measure all n ancilla qubits, decoder must compute syndrome
        current_meas_idx = measurement_offset
        
        # Find max qubit index to allocate ancillas after
        max_data_q = max(max(qs) for qs in data_qubits.values()) if data_qubits else -1
        ancilla_start = max_data_q + 1
        
        for block_id in sorted(data_qubits.keys()):
            data_qs = data_qubits[block_id]
            n = len(data_qs)
            
            # Allocate single ancilla block (reused for Z and X)
            ancilla_qs = list(range(ancilla_start, ancilla_start + n))
            ancilla_start += n
            
            block_verification_meas = []
            
            # === Z Syndrome Extraction (detects X errors on data) ===
            if self.extract_z_syndrome:
                # Prepare ancilla in |0⟩^n or |0_L⟩ (depending on ancilla_prep)
                if isinstance(self.ancilla_prep, VerifiedAncillaGadget):
                    prep_result = self.ancilla_prep.emit_prepare(
                        circuit=circuit,
                        ancilla_qubits=ancilla_qs,
                        basis=AncillaBasis.ZERO,
                        noise_model=noise_model,
                        measurement_offset=current_meas_idx,
                        post_select=effective_post_select,
                    )
                else:
                    prep_result = self.ancilla_prep.emit_prepare(
                        circuit=circuit,
                        ancilla_qubits=ancilla_qs,
                        basis=AncillaBasis.ZERO,
                        noise_model=noise_model,
                        measurement_offset=current_meas_idx,
                    )
                # Track verification measurements (if any)
                if prep_result.verification_measurements:
                    block_verification_meas.extend(prep_result.verification_measurements)
                current_meas_idx += prep_result.total_measurements
                
                circuit.append("TICK")
                
                # Transversal CNOT(data → ancilla)
                self.gate_strategy.emit_logical_cnot(circuit, data_qs, ancilla_qs, noise_model)
                circuit.append("TICK")
                
                # Measure ancilla in Z basis (use M not MR to avoid non-deterministic detectors)
                # The reset will happen at the start of the next ancilla preparation
                circuit.append("M", ancilla_qs)
                z_meas_indices = list(range(current_meas_idx, current_meas_idx + n))
                mmap.stabilizer_measurements['Z'][block_id] = z_meas_indices
                current_meas_idx += n
                circuit.append("TICK")
            
            # === X Syndrome Extraction (detects Z errors on data) ===
            if self.extract_x_syndrome:
                # Prepare ancilla in |+⟩^n or |+_L⟩ (depending on ancilla_prep)
                if isinstance(self.ancilla_prep, VerifiedAncillaGadget):
                    prep_result = self.ancilla_prep.emit_prepare(
                        circuit=circuit,
                        ancilla_qubits=ancilla_qs,
                        basis=AncillaBasis.PLUS,
                        noise_model=noise_model,
                        measurement_offset=current_meas_idx,
                        post_select=effective_post_select,
                    )
                else:
                    prep_result = self.ancilla_prep.emit_prepare(
                        circuit=circuit,
                        ancilla_qubits=ancilla_qs,
                        basis=AncillaBasis.PLUS,
                        noise_model=noise_model,
                        measurement_offset=current_meas_idx,
                    )
                # Track verification measurements (if any)
                if prep_result.verification_measurements:
                    block_verification_meas.extend(prep_result.verification_measurements)
                current_meas_idx += prep_result.total_measurements
                
                circuit.append("TICK")
                
                # Transversal CNOT(data → ancilla)
                self.gate_strategy.emit_logical_cnot(circuit, data_qs, ancilla_qs, noise_model)
                circuit.append("TICK")
                
                # Measure ancilla in X basis (H then M, not MR)
                # The reset will happen at the start of the next ancilla preparation
                self.gate_strategy.emit_logical_h(circuit, ancilla_qs, noise_model)
                circuit.append("M", ancilla_qs)
                x_meas_indices = list(range(current_meas_idx, current_meas_idx + n))
                mmap.stabilizer_measurements['X'][block_id] = x_meas_indices
                current_meas_idx += n
                circuit.append("TICK")
            
            # Store verification measurements for this block
            if block_verification_meas:
                mmap.verification_measurements[block_id] = block_verification_meas

        # If verification is present and expected to be post-selected, flag it
        if getattr(self.ancilla_prep, "uses_verification", False) and \
           getattr(self.ancilla_prep, "enable_post_selection", False) and \
           mmap.verification_measurements:
            mmap.requires_post_selection = True
        
        if isinstance(self.ancilla_prep, VerifiedAncillaGadget):
            mmap.requires_post_selection = bool(effective_post_select or mmap.verification_measurements)
            mmap.post_selection_detectors_added = bool(effective_post_select and mmap.verification_measurements)
            if effective_post_select and mmap.verification_measurements:
                mmap.verification_detectors = mmap.verification_measurements.copy()
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

