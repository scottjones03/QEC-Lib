"""
Stabilizer Syndrome Extraction Gadget.

This gadget performs standard stabilizer syndrome extraction as used in
most QEC memory experiments. For each stabilizer generator, it:
1. Prepares an ancilla qubit in |0⟩ (for Z stabilizers) or |+⟩ (for X stabilizers)
2. Applies CNOTs between the ancilla and the qubits in the stabilizer's support
3. Measures the ancilla to obtain the stabilizer eigenvalue

This is the standard syndrome extraction used in surface codes, Steane codes,
and other stabilizer codes for memory experiments.

NOTE: This is different from Steane's cat-state EC (SteaneECGadget), which uses
encoded ancilla blocks for fault-tolerant syndrome extraction.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, Optional, Any

import stim
import numpy as np

from .base import Gadget, MeasurementMap, SyndromeSchedule, LogicalMeasurementMap
from .noop_gadget import _get_z_support

if TYPE_CHECKING:
    from qectostim.codes.abstract_css import CSSCode
    from qectostim.noise.models import NoiseModel


class StabilizerSyndromeGadget(Gadget):
    """
    Standard stabilizer syndrome extraction gadget.
    
    For each stabilizer generator g_i with support S_i:
    - X stabilizer (row of Hx): Prepare ancilla in |+⟩, CNOT from ancilla to 
      each qubit in S_i, measure in X basis (H then M)
    - Z stabilizer (row of Hz): Prepare ancilla in |0⟩, CNOT from each qubit
      in S_i to ancilla, measure in Z basis
    
    This is the standard syndrome extraction used in most QEC experiments.
    
    Parameters
    ----------
    code : CSSCode
        The CSS code whose stabilizers to measure.
    extract_x : bool
        Whether to extract X stabilizer syndromes (detects Z errors).
    extract_z : bool
        Whether to extract Z stabilizer syndromes (detects X errors).
    """
    
    def __init__(
        self,
        code: Any,
        extract_x: bool = True,
        extract_z: bool = True,
    ):
        self.code = code
        self.extract_x = extract_x
        self.extract_z = extract_z
        
        # Get parity check matrices
        hx = code.hx
        hz = code.hz
        self.hx = np.atleast_2d(np.array(hx() if callable(hx) else hx, dtype=int))
        self.hz = np.atleast_2d(np.array(hz() if callable(hz) else hz, dtype=int))
        
        # Number of X and Z stabilizers
        self.n_x_stab = self.hx.shape[0] if self.extract_x else 0
        self.n_z_stab = self.hz.shape[0] if self.extract_z else 0
    
    @property
    def name(self) -> str:
        return "StabilizerSyndrome"
    
    @property
    def requires_ancillas(self) -> bool:
        return True
    
    @property
    def ancillas_per_block(self) -> int:
        """One ancilla per stabilizer generator."""
        return self.n_x_stab + self.n_z_stab
    
    def emit(
        self,
        circuit: stim.Circuit,
        data_qubits: Dict[int, List[int]],
        ancilla_qubits: Optional[Dict[int, List[int]]] = None,
        noise_model: Optional["NoiseModel"] = None,
        measurement_offset: int = 0,
    ) -> MeasurementMap:
        """
        Emit standard stabilizer syndrome extraction.
        
        For each block in data_qubits:
        1. X stabilizer extraction: For each X stabilizer (row of Hx),
           prepare ancilla in |+⟩, CNOT from ancilla to support, measure X
        2. Z stabilizer extraction: For each Z stabilizer (row of Hz),
           prepare ancilla in |0⟩, CNOT from support to ancilla, measure Z
        """
        def _apply_meas_noise(qs: List[int]) -> None:
            if not noise_model or not qs:
                return
            if hasattr(noise_model, 'apply_measurement_noise'):
                res = noise_model.apply_measurement_noise(qs)
                if res:
                    for inst in res:
                        circuit.append(inst)
            elif hasattr(noise_model, 'apply_single_qubit_noise'):
                for q in qs:
                    noise_model.apply_single_qubit_noise(circuit, q)

        def _count_measurements_local(c: stim.Circuit) -> int:
            cnt = 0
            for inst in c:
                if isinstance(inst, stim.CircuitRepeatBlock):
                    continue
                name = inst.name.upper()
                if name in {"M", "MR", "MX", "MY", "MZ", "MRX", "MRY", "MRZ"}:
                    cnt += len([t for t in inst.targets_copy() if t.is_qubit_target])
            return cnt

        pre_meas = _count_measurements_local(circuit)
        mmap = MeasurementMap(offset=measurement_offset)
        mmap.stabilizer_measurements = {'X': {}, 'Z': {}}
        current_meas = measurement_offset
        
        # Allocate ancillas after all data qubits
        max_data_q = max(max(qs) for qs in data_qubits.values())
        ancilla_start = max_data_q + 1
        
        for block_id, data_qs in sorted(data_qubits.items()):
            x_indices = []
            z_indices = []
            n_data = len(data_qs)
            
            # X stabilizer extraction (detects Z errors)
            if self.extract_x:
                for stab_idx in range(self.n_x_stab):
                    anc = ancilla_start
                    ancilla_start += 1
                    
                    # Get support of this X stabilizer
                    support = [data_qs[i] for i in range(n_data) 
                               if i < self.hx.shape[1] and self.hx[stab_idx, i] == 1]
                    
                    # Prepare ancilla in |+⟩
                    circuit.append("R", [anc])
                    circuit.append("H", [anc])
                    if noise_model:
                        noise_model.apply_single_qubit_noise(circuit, anc)
                    
                    # CNOT from each data qubit in support to ancilla
                    # This entangles ancilla with X parity of data
                    for q in support:
                        circuit.append("CX", [q, anc])  # CNOT(data → ancilla)
                        if noise_model:
                            noise_model.apply_two_qubit_noise(circuit, q, anc)
                    
                    # Measure in X basis
                    circuit.append("H", [anc])
                    if noise_model:
                        noise_model.apply_single_qubit_noise(circuit, anc)
                    circuit.append("M", [anc])
                    _apply_meas_noise([anc])
                    x_indices.append(current_meas)
                    current_meas += 1
            
            # Z stabilizer extraction (detects X errors)
            if self.extract_z:
                for stab_idx in range(self.n_z_stab):
                    anc = ancilla_start
                    ancilla_start += 1
                    
                    # Get support of this Z stabilizer
                    support = [data_qs[i] for i in range(n_data)
                               if i < self.hz.shape[1] and self.hz[stab_idx, i] == 1]
                    
                    # Prepare ancilla in |0⟩
                    circuit.append("R", [anc])
                    if noise_model:
                        noise_model.apply_single_qubit_noise(circuit, anc)
                    
                    # CNOT from each qubit in support to ancilla
                    for q in support:
                        circuit.append("CX", [q, anc])
                        if noise_model:
                            noise_model.apply_two_qubit_noise(circuit, q, anc)
                    
                    # Measure in Z basis
                    circuit.append("M", [anc])
                    _apply_meas_noise([anc])
                    z_indices.append(current_meas)
                    current_meas += 1
            
            mmap.stabilizer_measurements['X'][block_id] = x_indices
            mmap.stabilizer_measurements['Z'][block_id] = z_indices
            circuit.append("TICK")
        
        mmap.total_measurements = current_meas - measurement_offset
        post_meas = _count_measurements_local(circuit)
        if (post_meas - pre_meas) != mmap.total_measurements:
            raise AssertionError(
                f"Measurement count mismatch: expected {mmap.total_measurements}, got {post_meas - pre_meas}"
            )
        return mmap
    
    def get_syndrome_schedule(self) -> SyndromeSchedule:
        """Get syndrome extraction schedule."""
        stypes = []
        rounds = {}
        schedule = []
        
        if self.extract_x:
            stypes.append('X')
            rounds['X'] = 1
            schedule.append(('X', 0))
        
        if self.extract_z:
            stypes.append('Z')
            rounds['Z'] = 1
            schedule.append(('Z', 0))
        
        return SyndromeSchedule(
            stabilizer_types=stypes,
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
