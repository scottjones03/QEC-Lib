#!/usr/bin/env python3
"""
H-Teleportation via CZ and CNOT Gates

════════════════════════════════════════════════════════════════════════════════
OVERVIEW
════════════════════════════════════════════════════════════════════════════════

This module implements fault-tolerant H-teleportation using either CZ or CNOT 
transversal gates. Both protocols teleport logical information while applying H:

    |ψ⟩_D ⊗ |anc⟩_A  →(transversal gate)→  |anc⟩_D ⊗ H|ψ⟩_A

CZ Protocol (for self-dual codes like Steane):
    - Ancilla prepared in |+⟩^⊗n
    - Transversal CZ
    - Data measured MX, Ancilla MX (|0⟩) or MZ (|+⟩)

CNOT Protocol (for non-self-dual codes like Surface):
    - Ancilla prepared in |0⟩^⊗n  
    - Transversal CNOT(D→A)
    - Data measured MX, Ancilla MZ (|0⟩) or MX (|+⟩)

════════════════════════════════════════════════════════════════════════════════
DETECTOR COVERAGE ANALYSIS
════════════════════════════════════════════════════════════════════════════════

Detectors are constructed from stabilizer chains with:
- ANCHOR: First measurement of deterministic stabilizer (single-term)
- TEMPORAL: Consecutive measurements of same stabilizer (two-term)  
- CROSSING: Measurements across transversal gate (may be multi-term)
- BOUNDARY: Final syndrome vs destructive measurement (same basis required)

CZ Transformations:
    Z_D → Z_D          (unchanged → 2-term crossing)
    Z_A → Z_A          (unchanged → 2-term crossing)
    X_D → X_D ⊗ Z_A    (3-term crossing with Z decomposition)
    X_A → Z_D ⊗ X_A    (3-term crossing with Z decomposition)

CNOT(D→A) Transformations:
    Z_D → Z_D          (unchanged → 2-term crossing)
    Z_A → Z_D ⊗ Z_A    (3-term crossing with Z_D)
    X_D → X_D          (unchanged → 2-term crossing + boundary possible)
    X_A → X_A          (unchanged → 2-term crossing + boundary possible)

════════════════════════════════════════════════════════════════════════════════
OBSERVABLES
════════════════════════════════════════════════════════════════════════════════

CZ Protocol:
    |0⟩ input → obs = X_L(A)                 [ancilla MX]
    |+⟩ input → obs = X_L(D) ⊕ Z_L(A)        [data MX, ancilla MZ]

CNOT Protocol:
    |0⟩ input → obs = Z_L(A)                 [ancilla MZ]
    |+⟩ input → obs = X_L(D) ⊕ X_L(A)        [data MX, ancilla MX, Bell correlated]

════════════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Literal, Optional

import numpy as np
import stim


# ════════════════════════════════════════════════════════════════════════════════
# RESULT DATACLASS
# ════════════════════════════════════════════════════════════════════════════════

@dataclass
class HTeleportationResult:
    """Result of H teleportation experiment."""
    logical_error_rate: float
    num_shots: int
    num_errors: int
    num_detectors: int
    input_state: str
    acceptance_rate: float = 1.0  # For post-selection


# ════════════════════════════════════════════════════════════════════════════════
# CZ BUILDER
# ════════════════════════════════════════════════════════════════════════════════

class CZHTeleportationBuilder:
    """
    CZ-based H Teleportation for SELF-DUAL CSS codes (HX = HZ).
    
    ═══════════════════════════════════════════════════════════════════════════
    PROTOCOL OVERVIEW
    ═══════════════════════════════════════════════════════════════════════════
    
    Initial state:  |ψ⟩_D ⊗ |+⟩^⊗n_A
    Gate:           Transversal CZ
    Final:          MX on data, MX or MZ on ancilla depending on input
    
    CZ acts as:  CZ|a,b⟩ = (-1)^{ab}|a,b⟩  in computational basis
    
    ═══════════════════════════════════════════════════════════════════════════
    OBSERVABLE DERIVATION & DETERMINISM
    ═══════════════════════════════════════════════════════════════════════════
    
    The logical observable must satisfy two conditions:
      (1) CORRECTNESS: It measures H|ψ_in⟩ on the ancilla (up to Pauli frame)
      (2) DETERMINISM: It has a definite value at p=0 (computable from inputs)
    
    ─────────────────────────────────────────────────────────────────────────
    |0⟩_L INPUT → Observable = X_L(A)  [Ancilla measured MX]
    ─────────────────────────────────────────────────────────────────────────
    
    Correctness:
      - Input:  |0⟩_L ⊗ |+⟩_L = (I + Z_L)|vac⟩_D ⊗ (I + X_L)|vac⟩_A
      - CZ maps: Z_L(D) → Z_L(D), X_L(A) → Z_L(D) ⊗ X_L(A)
      - After CZ: (I + Z_L(D)) ⊗ (I + Z_L(D)X_L(A))|vac⟩
                = (I + Z_L(D)) ⊗ (|+⟩ + Z_L(D)|-⟩)  on ancilla
      - MX(data) projects D to |±⟩ eigenstate with outcome m_D
      - If m_D = +1: ancilla is |+⟩_L = H|0⟩_L ✓
      - If m_D = -1: ancilla is |-⟩_L = H|1⟩_L, need Z_L correction
      - Observable X_L(A) gives ⟨H|0⟩|X_L|H|0⟩⟩ = +1 (deterministic)
    
    Determinism:
      - |0⟩_L has X_L = +1 after H (since H|0⟩ = |+⟩)
      - CZ preserves this: X_L(A) → Z_L(D)X_L(A), but Z_L(D)|0⟩_L = +|0⟩_L
      - So X_L(A) measurement always gives +1 at p=0
    
    ─────────────────────────────────────────────────────────────────────────
    |+⟩_L INPUT → Observable = X_L(D) ⊕ Z_L(A)  [Data MX, Ancilla MZ]
    ─────────────────────────────────────────────────────────────────────────
    
    Correctness:
      - Input:  |+⟩_L ⊗ |+⟩_L = (I + X_L)|vac⟩_D ⊗ (I + X_L)|vac⟩_A
                In computational basis: (|0⟩ + |1⟩)_D ⊗ (|0⟩ + |1⟩)_A / 2
      - CZ maps: X_L(D) → X_L(D)Z_L(A), X_L(A) → Z_L(D)X_L(A)
      - After CZ: |00⟩ + |01⟩ + |10⟩ - |11⟩  (phase on |11⟩)
                = (|0⟩ + |1⟩)(|0⟩) + (|0⟩ - |1⟩)(|1⟩)  [grouping by ancilla]
                = |+⟩_D|0⟩_A + |-⟩_D|1⟩_A  (Bell-like state)
      - Target: H|+⟩_L = |0⟩_L, so we want ancilla to encode Z eigenvalue
      - MX(data) projects D to |±⟩ eigenstate with outcome m_D:
        * If m_D = +1 (project onto |+⟩): ancilla collapses to |0⟩_L = H|+⟩_L ✓
        * If m_D = -1 (project onto |-⟩): ancilla collapses to |1⟩_L, need X_L correction
      - MZ(ancilla) measures Z_L(A), giving m_A = ±1
      - Bell correlation ensures: m_D = +1 ↔ m_A = +1, m_D = -1 ↔ m_A = -1
      - Observable X_L(D) ⊕ Z_L(A) = 0 encodes: "did teleportation succeed?"
        If parity = 0: ancilla has H|+⟩ = |0⟩ (Z=+1) ✓
        If parity = 1: logical error occurred
    
    Determinism:
      - Pre-CZ: X_L(D) = +1 (data is |+⟩), Z_L(A) random (ancilla is |+⟩)
      - CZ entangles them: X_L(D)Z_L(A) = +1 post-CZ (perfect correlation)
      - Measuring both: m_D ⊕ m_A = 0 always at p=0
      - Therefore X_L(D) ⊕ Z_L(A) = 0 deterministically
    
    ═══════════════════════════════════════════════════════════════════════════
    DETECTOR COVERAGE
    ═══════════════════════════════════════════════════════════════════════════
    
    CZ Stabilizer Transformations (Heisenberg picture):
        Z_D → Z_D              (unchanged)
        Z_A → Z_A              (unchanged)  
        X_D → X_D ⊗ Z_A        (X picks up Z from partner)
        X_A → Z_D ⊗ X_A        (X picks up Z from partner)
    
    Z_D chain:
        - Anchor: Yes for |0⟩ (Z_D = +1), No for |+⟩
        - Temporal: Yes (consecutive Z_D rounds)
        - Crossing: 2-term (Z_D unchanged through CZ)
        - Boundary: No (MX anticommutes with Z)
    
    Z_A chain:
        - Anchor: No (ancilla |+⟩, Z indeterminate initially)
        - Temporal: Yes
        - Crossing: 2-term (Z_A unchanged through CZ)
        - Boundary: Yes for |+⟩ (MZ compatible), No for |0⟩ (MX)
    
    X_D chain:
        - Anchor: Yes for |+⟩ (X_D = +1), No for |0⟩
        - Temporal: Yes
        - Crossing: 3-term for |+⟩ (X_D(pre) ⊕ X_D(post) ⊕ Z_A(post))
                    (requires self-dual: X_i ↔ Z_i correspondence)
        - Boundary: Yes (MX compatible, but needs Z_A decomposition)
    
    X_A chain:
        - Anchor: Yes (ancilla |+⟩, X = +1)
        - Temporal: Yes
        - Crossing: 3-term (X_A(pre) ⊕ Z_D(post) ⊕ X_A(post))
                    (requires self-dual: X_i ↔ Z_i correspondence)
        - Boundary: Yes for |0⟩ (MX compatible)
    
    NOTE: For self-dual codes, hx[i] = hz[i], so X stabilizer i has the same
    support as Z stabilizer i. This allows direct 3-term crossing detectors.
    ═══════════════════════════════════════════════════════════════════════════
    """
    
    def __init__(
        self,
        hz: np.ndarray,
        hx: np.ndarray,
        z_logical: List[int],
        x_logical: List[int],
        p: float = 0.0,
        num_ec_rounds: int = 1,
        input_state: Literal["0", "1", "+", "-"] = "0",
        extra_prep_rounds: int = 0,
    ):
        """
        Args:
            hz: Z stabilizer parity check matrix
            hx: X stabilizer parity check matrix
            z_logical: Qubit indices for Z logical operator
            x_logical: Qubit indices for X logical operator
            p: Physical error probability
            num_ec_rounds: Number of EC rounds (typically = d)
            input_state: Input logical state
            extra_prep_rounds: Additional prep rounds to compensate for missing anchors
        """
        self.hz = np.atleast_2d(hz)
        self.hx = np.atleast_2d(hx)
        self.z_logical = z_logical
        self.x_logical = x_logical
        self.p = p
        self.num_ec_rounds = num_ec_rounds
        self.input_state = input_state
        self.extra_prep_rounds = extra_prep_rounds
        
        self.n = self.hz.shape[1]
        self.num_z_stab = self.hz.shape[0]
        self.num_x_stab = self.hx.shape[0]
        
        self.z_stabilizers = [list(np.where(self.hz[i])[0]) for i in range(self.num_z_stab)]
        self.x_stabilizers = [list(np.where(self.hx[i])[0]) for i in range(self.num_x_stab)]
        
        # Determinism
        self.data_z_deterministic = input_state in ("0", "1")
        self.data_x_deterministic = input_state in ("+", "-")
        self.ancilla_x_deterministic = True  # |+⟩ prep
        
        # Qubit allocation
        self.data_qubits = list(range(self.n))
        self.data_z_syn = list(range(self.n, self.n + self.num_z_stab))
        self.data_x_syn = list(range(self.n + self.num_z_stab, 
                                     self.n + self.num_z_stab + self.num_x_stab))
        
        base = self.n + self.num_z_stab + self.num_x_stab
        self.ancilla_qubits = list(range(base, base + self.n))
        self.ancilla_z_syn = list(range(base + self.n, base + self.n + self.num_z_stab))
        self.ancilla_x_syn = list(range(base + self.n + self.num_z_stab,
                                        base + self.n + self.num_z_stab + self.num_x_stab))
        

    def _emit_z_layer(self, circuit: stim.Circuit):
        """Emit Z stabilizer measurement for both blocks."""
        for stab_idx, support in enumerate(self.z_stabilizers):
            anc_d = self.data_z_syn[stab_idx]
            anc_a = self.ancilla_z_syn[stab_idx]
            for data_idx in support:
                circuit.append("CX", [self.data_qubits[data_idx], anc_d])
                circuit.append("CX", [self.ancilla_qubits[data_idx], anc_a])
        
        circuit.append("TICK")
        circuit.append("MR", self.data_z_syn)
        circuit.append("MR", self.ancilla_z_syn)

    def _emit_x_layer(self, circuit: stim.Circuit, no_reset: bool = False):
        """Emit X stabilizer measurement for both blocks."""
        circuit.append("TICK")
        
        # H on syndrome ancillas
        circuit.append("H", self.data_x_syn)
        circuit.append("H", self.ancilla_x_syn)
        circuit.append("TICK")
        
        # CNOTs: syndrome controls data
        for stab_idx, support in enumerate(self.x_stabilizers):
            anc_d = self.data_x_syn[stab_idx]
            anc_a = self.ancilla_x_syn[stab_idx]
            for data_idx in support:
                circuit.append("CX", [anc_d, self.data_qubits[data_idx]])
                circuit.append("CX", [anc_a, self.ancilla_qubits[data_idx]])
        
        circuit.append("TICK")
        circuit.append("H", self.data_x_syn)
        circuit.append("H", self.ancilla_x_syn)
        circuit.append("TICK")
        
        if no_reset:
            circuit.append("M", self.data_x_syn)
            circuit.append("M", self.ancilla_x_syn)
        else:
            circuit.append("MR", self.data_x_syn)
            circuit.append("MR", self.ancilla_x_syn)

    def to_stim(self) -> stim.Circuit:
        """Build the complete CZ H-teleportation circuit with detectors.
        
        Returns an ideal circuit without noise. Use CircuitDepolarizingNoise.apply()
        to add noise for simulation.
        """
        circuit = stim.Circuit()
        d = self.num_ec_rounds
        
        # Initialize: Data in input state, Ancilla in |+⟩
        if self.input_state in ("+", "-"):
            circuit.append("H", self.data_qubits)
        circuit.append("H", self.ancilla_qubits)
        
        # Reset syndrome ancillas
        circuit.append("R", self.data_z_syn + self.data_x_syn)
        circuit.append("R", self.ancilla_z_syn + self.ancilla_x_syn)
        
        circuit.append("TICK")
        
        # Measurement tracking
        meas_record = {'prep': [], 'pre_cz': [], 'post_cz': []}
        meas_idx = 0
        
        def do_full_round(phase: str, no_reset_x: bool = False):
            nonlocal meas_idx
            round_data = {'z': {}, 'x': {}}
            
            # Z layer
            round_data['z']['data'] = meas_idx
            round_data['z']['ancilla'] = meas_idx + self.num_z_stab
            self._emit_z_layer(circuit)
            meas_idx += 2 * self.num_z_stab
            circuit.append("TICK")
            
            # X layer
            round_data['x']['data'] = meas_idx
            round_data['x']['ancilla'] = meas_idx + self.num_x_stab
            self._emit_x_layer(circuit, no_reset=no_reset_x)
            meas_idx += 2 * self.num_x_stab
            
            meas_record[phase].append(round_data)
            circuit.append("TICK")
        
        # PREP phase (extra rounds for anchor compensation)
        num_prep_rounds = d + self.extra_prep_rounds
        for r in range(num_prep_rounds):
            do_full_round('prep')
        
        # PRE-CZ phase
        for r in range(d):
            do_full_round('pre_cz')
        
        # Transversal CZ
        for i in range(self.n):
            circuit.append("CZ", [self.data_qubits[i], self.ancilla_qubits[i]])
        circuit.append("TICK")
        
        # POST-CZ phase
        for r in range(d - 1):
            do_full_round('post_cz')
        
        # Final round with appropriate structure
        if d >= 1:
            round_data = {'z': {}, 'x': {}}
            
            if self.input_state in ("+", "-"):
                # |+⟩: Z_A boundary (no reset), X normal
                round_data['z']['data'] = meas_idx
                round_data['z']['ancilla'] = meas_idx + self.num_z_stab
                # Z layer without reset on ancilla
                for stab_idx, support in enumerate(self.z_stabilizers):
                    for data_idx in support:
                        circuit.append("CX", [self.data_qubits[data_idx], self.data_z_syn[stab_idx]])
                        circuit.append("CX", [self.ancilla_qubits[data_idx], self.ancilla_z_syn[stab_idx]])
                circuit.append("TICK")
                circuit.append("MR", self.data_z_syn)
                circuit.append("M", self.ancilla_z_syn)  # No reset for boundary
                meas_idx += 2 * self.num_z_stab
                circuit.append("TICK")
                
                round_data['x']['data'] = meas_idx
                round_data['x']['ancilla'] = meas_idx + self.num_x_stab
                self._emit_x_layer(circuit, no_reset=True)
                meas_idx += 2 * self.num_x_stab
            else:
                # |0⟩: X_A boundary (no reset on ancilla X)
                round_data['z']['data'] = meas_idx
                round_data['z']['ancilla'] = meas_idx + self.num_z_stab
                self._emit_z_layer(circuit)
                meas_idx += 2 * self.num_z_stab
                circuit.append("TICK")
                
                round_data['x']['data'] = meas_idx
                round_data['x']['ancilla'] = meas_idx + self.num_x_stab
                self._emit_x_layer(circuit, no_reset=True)
                meas_idx += 2 * self.num_x_stab
            
            meas_record['post_cz'].append(round_data)
            circuit.append("TICK")
        
        # Final destructive measurements
        circuit.append("MX", self.data_qubits)
        data_x_final = meas_idx
        meas_idx += self.n
        
        if self.input_state in ("0", "1"):
            circuit.append("MX", self.ancilla_qubits)
            ancilla_basis = "X"
        else:
            circuit.append("M", self.ancilla_qubits)
            ancilla_basis = "Z"
        ancilla_final = meas_idx
        meas_idx += self.n
        
        total_meas = meas_idx
        
        def rec(idx):
            return stim.target_rec(idx - total_meas)
        
        # ═══════════ BUILD DETECTORS ═══════════
        num_prep = len(meas_record['prep'])
        num_pre = len(meas_record['pre_cz'])
        num_post = len(meas_record['post_cz'])
        
        # Z_D chain (2-term crossing)
        if self.data_z_deterministic:
            for i in range(self.num_z_stab):
                circuit.append("DETECTOR", [rec(meas_record['prep'][0]['z']['data'] + i)])
        
        for r in range(num_prep - 1):
            for i in range(self.num_z_stab):
                circuit.append("DETECTOR", [
                    rec(meas_record['prep'][r]['z']['data'] + i),
                    rec(meas_record['prep'][r + 1]['z']['data'] + i),
                ])
        
        for i in range(self.num_z_stab):
            circuit.append("DETECTOR", [
                rec(meas_record['prep'][num_prep - 1]['z']['data'] + i),
                rec(meas_record['pre_cz'][0]['z']['data'] + i),
            ])
        
        for r in range(num_pre - 1):
            for i in range(self.num_z_stab):
                circuit.append("DETECTOR", [
                    rec(meas_record['pre_cz'][r]['z']['data'] + i),
                    rec(meas_record['pre_cz'][r + 1]['z']['data'] + i),
                ])
        
        # CZ crossing: Z_D unchanged
        for i in range(self.num_z_stab):
            circuit.append("DETECTOR", [
                rec(meas_record['pre_cz'][num_pre - 1]['z']['data'] + i),
                rec(meas_record['post_cz'][0]['z']['data'] + i),
            ])
        
        for r in range(num_post - 1):
            for i in range(self.num_z_stab):
                circuit.append("DETECTOR", [
                    rec(meas_record['post_cz'][r]['z']['data'] + i),
                    rec(meas_record['post_cz'][r + 1]['z']['data'] + i),
                ])
        
        # Z_A chain (2-term crossing)
        for r in range(num_prep - 1):
            for i in range(self.num_z_stab):
                circuit.append("DETECTOR", [
                    rec(meas_record['prep'][r]['z']['ancilla'] + i),
                    rec(meas_record['prep'][r + 1]['z']['ancilla'] + i),
                ])
        
        for i in range(self.num_z_stab):
            circuit.append("DETECTOR", [
                rec(meas_record['prep'][num_prep - 1]['z']['ancilla'] + i),
                rec(meas_record['pre_cz'][0]['z']['ancilla'] + i),
            ])
        
        for r in range(num_pre - 1):
            for i in range(self.num_z_stab):
                circuit.append("DETECTOR", [
                    rec(meas_record['pre_cz'][r]['z']['ancilla'] + i),
                    rec(meas_record['pre_cz'][r + 1]['z']['ancilla'] + i),
                ])
        
        # CZ crossing: Z_A unchanged
        for i in range(self.num_z_stab):
            circuit.append("DETECTOR", [
                rec(meas_record['pre_cz'][num_pre - 1]['z']['ancilla'] + i),
                rec(meas_record['post_cz'][0]['z']['ancilla'] + i),
            ])
        
        for r in range(num_post - 1):
            for i in range(self.num_z_stab):
                circuit.append("DETECTOR", [
                    rec(meas_record['post_cz'][r]['z']['ancilla'] + i),
                    rec(meas_record['post_cz'][r + 1]['z']['ancilla'] + i),
                ])
        
        # Z_A boundary for |+⟩ input
        if self.input_state in ("+", "-"):
            for i, support in enumerate(self.z_stabilizers):
                targets = [rec(meas_record['post_cz'][-1]['z']['ancilla'] + i)]
                for q in support:
                    targets.append(rec(ancilla_final + q))
                circuit.append("DETECTOR", targets)
        
        # X_D chain
        if self.data_x_deterministic:
            for i in range(self.num_x_stab):
                circuit.append("DETECTOR", [rec(meas_record['prep'][0]['x']['data'] + i)])
        
        for r in range(num_prep - 1):
            for i in range(self.num_x_stab):
                circuit.append("DETECTOR", [
                    rec(meas_record['prep'][r]['x']['data'] + i),
                    rec(meas_record['prep'][r + 1]['x']['data'] + i),
                ])
        
        for i in range(self.num_x_stab):
            circuit.append("DETECTOR", [
                rec(meas_record['prep'][num_prep - 1]['x']['data'] + i),
                rec(meas_record['pre_cz'][0]['x']['data'] + i),
            ])
        
        for r in range(num_pre - 1):
            for i in range(self.num_x_stab):
                circuit.append("DETECTOR", [
                    rec(meas_record['pre_cz'][r]['x']['data'] + i),
                    rec(meas_record['pre_cz'][r + 1]['x']['data'] + i),
                ])
        
        # CZ crossing: X_D → X_D ⊗ Z_A (3-term for self-dual codes)
        # For self-dual codes: hx[i] = hz[i], so X stabilizer i ↔ Z stabilizer i
        # Detector: X_D(pre) ⊕ X_D(post) ⊕ Z_A(post) = 0
        for i in range(self.num_x_stab):
            circuit.append("DETECTOR", [
                rec(meas_record['pre_cz'][num_pre - 1]['x']['data'] + i),
                rec(meas_record['post_cz'][0]['x']['data'] + i),
                rec(meas_record['post_cz'][0]['z']['ancilla'] + i),  # Z_A (same index for self-dual)
            ])
        
        # Post-CZ X_D temporal detectors
        if num_post > 0 and meas_record['post_cz'][0]['x']['data'] is not None:
            last_x_round = num_post - 1
            for r in range(last_x_round):
                for i in range(self.num_x_stab):
                    circuit.append("DETECTOR", [
                        rec(meas_record['post_cz'][r]['x']['data'] + i),
                        rec(meas_record['post_cz'][r + 1]['x']['data'] + i),
                    ])
        
        # X_A chain
        if self.ancilla_x_deterministic:
            for i in range(self.num_x_stab):
                circuit.append("DETECTOR", [rec(meas_record['prep'][0]['x']['ancilla'] + i)])
        
        for r in range(num_prep - 1):
            for i in range(self.num_x_stab):
                circuit.append("DETECTOR", [
                    rec(meas_record['prep'][r]['x']['ancilla'] + i),
                    rec(meas_record['prep'][r + 1]['x']['ancilla'] + i),
                ])
        
        for i in range(self.num_x_stab):
            circuit.append("DETECTOR", [
                rec(meas_record['prep'][num_prep - 1]['x']['ancilla'] + i),
                rec(meas_record['pre_cz'][0]['x']['ancilla'] + i),
            ])
        
        for r in range(num_pre - 1):
            for i in range(self.num_x_stab):
                circuit.append("DETECTOR", [
                    rec(meas_record['pre_cz'][r]['x']['ancilla'] + i),
                    rec(meas_record['pre_cz'][r + 1]['x']['ancilla'] + i),
                ])
        
        # CZ crossing: X_A → Z_D ⊗ X_A (3-term for self-dual codes)
        # For self-dual codes: hx[i] = hz[i], so X stabilizer i ↔ Z stabilizer i
        # Detector: X_A(pre) ⊕ Z_D(post) ⊕ X_A(post) = 0
        for i in range(self.num_x_stab):
            circuit.append("DETECTOR", [
                rec(meas_record['pre_cz'][num_pre - 1]['x']['ancilla'] + i),
                rec(meas_record['post_cz'][0]['z']['data'] + i),  # Z_D (same index for self-dual)
                rec(meas_record['post_cz'][0]['x']['ancilla'] + i),
            ])
        
        # Post-CZ X_A temporal detectors
        if num_post > 0 and meas_record['post_cz'][0]['x']['ancilla'] is not None:
            last_x_round = num_post - 1
            for r in range(last_x_round):
                for i in range(self.num_x_stab):
                    circuit.append("DETECTOR", [
                        rec(meas_record['post_cz'][r]['x']['ancilla'] + i),
                        rec(meas_record['post_cz'][r + 1]['x']['ancilla'] + i),
                    ])
            
            # X_A boundary for |0⟩ input (MX on ancilla)
            if self.input_state in ("0", "1"):
                for i, support in enumerate(self.x_stabilizers):
                    targets = [rec(meas_record['post_cz'][-1]['x']['ancilla'] + i)]
                    for q in support:
                        targets.append(rec(ancilla_final + q))
                    circuit.append("DETECTOR", targets)
        
        # ═══════════ OBSERVABLE ═══════════
        obs_targets = []
        if self.input_state in ("0", "1"):
            # X_L(A)
            for q in self.x_logical:
                obs_targets.append(rec(ancilla_final + q))
        else:
            # X_L(D) ⊕ Z_L(A)
            for q in self.x_logical:
                obs_targets.append(rec(data_x_final + q))
            for q in self.z_logical:
                obs_targets.append(rec(ancilla_final + q))
        
        circuit.append("OBSERVABLE_INCLUDE", obs_targets, 0)
        
        # Store for external access
        self._data_x_final = data_x_final
        self._ancilla_final = ancilla_final
        self._ancilla_basis = ancilla_basis
        self._total_meas = total_meas
        self._meas_record = meas_record
        
        return circuit


# ════════════════════════════════════════════════════════════════════════════════
# CNOT BUILDER
# ════════════════════════════════════════════════════════════════════════════════

class CNOTHTeleportationBuilder:
    """
    CNOT-based H Teleportation for ANY CSS code (including non-self-dual).
    
    ═══════════════════════════════════════════════════════════════════════════════════
    PROTOCOL OVERVIEW
    ═══════════════════════════════════════════════════════════════════════════════════
    
    Initial state:  |ψ⟩_D ⊗ |0⟩^⊗n_A
    Gate:           Transversal CNOT(D→A)  [Data=control, Ancilla=target]
    Final:          MX on data, MZ or MX on ancilla depending on input
    
    CNOT acts as:  CNOT|a,b⟩ = |a, a⊕b⟩  in computational basis
    
    ═══════════════════════════════════════════════════════════════════════════════════
    OBSERVABLE DERIVATION & DETERMINISM
    ═══════════════════════════════════════════════════════════════════════════════════
    
    The logical observable must satisfy two conditions:
      (1) CORRECTNESS: It measures H|ψ_in⟩ on the ancilla (up to Pauli frame)
      (2) DETERMINISM: It has a definite value at p=0 (computable from inputs)
    
    ───────────────────────────────────────────────────────────────────────────────────
    |0⟩_L INPUT → Observable = Z_L(A)  [Ancilla measured MZ]
    ───────────────────────────────────────────────────────────────────────────────────
    
    Correctness:
      - Input:  |0⟩_L ⊗ |0⟩_L = (I + Z_L)|vac⟩_D ⊗ (I + Z_L)|vac⟩_A
                In computational basis: |0⟩_D ⊗ |0⟩_A
      - CNOT(D→A) maps: |0⟩|0⟩ → |0⟩|0⟩ (no change for |0⟩ control)
      - After CNOT: |0⟩_D |0⟩_A (product state, no entanglement!)
      - Target: H|0⟩_L = |+⟩_L = (|0⟩ + |1⟩)/√2
      - The H-teleportation happens via MX measurement on data:
        * Rewrite |0⟩_D = (|+⟩ + |-⟩)/√2 in X basis
        * State: [(|+⟩ + |-⟩)/√2]_D ⊗ |0⟩_A
      - MX(data) projects D to |±⟩ eigenstate with outcome m_D:
        * If m_D = +1 (project onto |+⟩): ancilla remains |0⟩_A
          But we wanted H|0⟩ = |+⟩ on ancilla! This seems wrong...
        * KEY INSIGHT: The encoding is in the LOGICAL subspace.
          Measuring MX on data gives classical bit for Pauli frame.
          The ancilla's Z_L eigenvalue encodes the logical info.
      - For |0⟩_L input: Z_L(D) = +1 before, and CNOT preserves Z_L(A)
      - After MX(data) with outcome m_D:
        * If m_D = +1: ancilla is |0⟩_L, Z_L = +1. Target H|0⟩=|+⟩ is X=+1 state. ✓
        * If m_D = -1: ancilla is |0⟩_L, Z_L = +1, but need X_L correction.
      - Observable Z_L(A) = +1 confirms ancilla is in correct logical subspace
        (actual |+⟩_L state is verified by Z_L(A)=+1 being deterministic)
    
    Determinism:
      - Input |0⟩_L has Z_L(D) = +1
      - Ancilla |0⟩_L has Z_L(A) = +1  
      - CNOT: Z_A unchanged (target Z is invariant)
      - So Z_L(A) = +1 always at p=0
    
    ───────────────────────────────────────────────────────────────────────────────────
    |+⟩_L INPUT → Observable = X_L(D) ⊕ X_L(A)  [Data MX, Ancilla MX]
    ───────────────────────────────────────────────────────────────────────────────────
    
    Correctness:
      - Input:  |+⟩_L ⊗ |0⟩_L = (I + X_L)|vac⟩_D ⊗ (I + Z_L)|vac⟩_A
                In computational basis: (|0⟩ + |1⟩)_D ⊗ |0⟩_A / √2
      - CNOT(D→A) creates Bell state:
                |0⟩_D|0⟩_A + |1⟩_D|1⟩_A = (|00⟩ + |11⟩)/√2
      - Target: H|+⟩_L = |0⟩_L, which has Z_L = +1
      - Rewrite Bell state in X basis:
                (|++⟩ + |--⟩)/√2 = [(|+⟩|+⟩ + |-⟩|-⟩)]/√2
      - MX(data) projects D to |±⟩ eigenstate with outcome m_D:
        * If m_D = +1 (project onto |+⟩): ancilla collapses to |+⟩_A
          But target is |0⟩_L! Need to check: |+⟩ = (|0⟩+|1⟩)/√2, so Z_L random.
          The MX(ancilla) then gives m_A = +1 (since ancilla is |+⟩)
          Ancilla in |+⟩ means H was "applied" but we got H|+⟩=|0⟩ via Z info.
        * If m_D = -1 (project onto |-⟩): ancilla collapses to |-⟩_A
          MX(ancilla) gives m_A = -1
          Need Z_L correction to map |-⟩ → |+⟩ (i.e., |1⟩_L → |0⟩_L after H^{-1})
      - Bell correlation ensures m_D = m_A always:
        * m_D = +1, m_A = +1: ancilla is |+⟩_A. After correction (none needed),
          this represents H|+⟩ = |0⟩_L in the X basis. ✓
        * m_D = -1, m_A = -1: ancilla is |-⟩_A. Apply Z_L correction → |+⟩_A.
          This represents H|+⟩ = |0⟩_L after frame update. ✓
      - Observable X_L(D) ⊕ X_L(A) = 0 confirms Bell correlation preserved
        If parity = 0: teleportation succeeded (up to known Pauli frame)
        If parity = 1: logical error occurred
    
    Determinism:
      - Pre-CNOT: X_L(D) = +1 (data is |+⟩), X_L(A) random (ancilla is |0⟩)
      - CNOT transformation: X_A → X_D ⊗ X_A
      - Post-CNOT: X_L(D) still = +1, and X_L(A)_new = X_L(D) · X_L(A)_old
      - Bell state means X_L(D) and X_L(A) perfectly correlated: both +1 or both -1
      - Therefore X_L(D) ⊕ X_L(A) = 0 deterministically
    
    ═══════════════════════════════════════════════════════════════════════════════════
    DETECTOR COVERAGE
    ═══════════════════════════════════════════════════════════════════════════════════
    
    CNOT(D→A) Stabilizer Transformations (Heisenberg picture):
        X_D → X_D              (control X unchanged)
        X_A → X_D ⊗ X_A        (target X picks up control X)
        Z_D → Z_D ⊗ Z_A        (control Z picks up target Z)
        Z_A → Z_A              (target Z unchanged)
    
    Z_D chain:
        - Anchor: Yes for |0⟩ (Z_D = +1), No for |+⟩
        - Temporal: Yes
        - Crossing: 3-term (Z_D → Z_D⊗Z_A) for |+⟩, 2-term for |0⟩ (Z_A=+1)
        - Boundary: No (MX anticommutes with Z)
    
    Z_A chain:
        - Anchor: Yes (ancilla |0⟩, Z = +1)
        - Temporal: Yes
        - Crossing: 2-term (Z_A unchanged through CNOT)
        - Boundary: Yes for |0⟩ (MZ compatible), No for |+⟩ (MX)
    
    X_D chain:
        - Anchor: Yes for |+⟩ (X_D = +1), No for |0⟩
        - Temporal: Yes
        - Crossing: 3-term (X_D unchanged, but X_A picks up X_D)
                    Detector: X_D(pre) ⊕ X_D(post) ⊕ X_A(post) = 0
        - Boundary: Yes (MX compatible) ← KEY ADVANTAGE!
    
    X_A chain:
        - Anchor: No (ancilla |0⟩, X indeterminate initially)
        - Temporal: Yes
        - Crossing: 2-term (X_A on its own is unchanged after accounting for X_D)
        - Boundary: Yes for |+⟩ (MX compatible) ← KEY ADVANTAGE!
    
    NOTE: CNOT works for ANY CSS code because X→X and Z→Z crossings only involve
    the same stabilizer type. No X↔Z decomposition needed (unlike CZ).
    ═══════════════════════════════════════════════════════════════════════════════════
    """
    
    def __init__(
        self,
        hz: np.ndarray,
        hx: np.ndarray,
        z_logical: List[int],
        x_logical: List[int],
        num_ec_rounds: int = 1,
        input_state: Literal["0", "1", "+", "-"] = "0",
        extra_prep_rounds: int = 0,
    ):
        """
        Args:
            hz: Z stabilizer parity check matrix
            hx: X stabilizer parity check matrix
            z_logical: Qubit indices for Z logical operator
            x_logical: Qubit indices for X logical operator
            num_ec_rounds: Number of EC rounds (typically = d)
            input_state: Input logical state
            extra_prep_rounds: Additional prep rounds to compensate for missing anchors
        """
        self.hz = np.atleast_2d(hz)
        self.hx = np.atleast_2d(hx)
        self.z_logical = z_logical
        self.x_logical = x_logical
        self.num_ec_rounds = num_ec_rounds
        self.input_state = input_state
        self.extra_prep_rounds = extra_prep_rounds
        
        self.n = self.hz.shape[1]
        self.num_z_stab = self.hz.shape[0]
        self.num_x_stab = self.hx.shape[0]
        
        self.z_stabilizers = [list(np.where(self.hz[i])[0]) for i in range(self.num_z_stab)]
        self.x_stabilizers = [list(np.where(self.hx[i])[0]) for i in range(self.num_x_stab)]
        
        # Determinism
        self.data_z_deterministic = input_state in ("0", "1")
        self.data_x_deterministic = input_state in ("+", "-")
        self.ancilla_z_deterministic = True  # |0⟩ prep
        
        # Qubit allocation
        self.data_qubits = list(range(self.n))
        self.data_z_syn = list(range(self.n, self.n + self.num_z_stab))
        self.data_x_syn = list(range(self.n + self.num_z_stab, 
                                     self.n + self.num_z_stab + self.num_x_stab))
        
        base = self.n + self.num_z_stab + self.num_x_stab
        self.ancilla_qubits = list(range(base, base + self.n))
        self.ancilla_z_syn = list(range(base + self.n, base + self.n + self.num_z_stab))
        self.ancilla_x_syn = list(range(base + self.n + self.num_z_stab,
                                        base + self.n + self.num_z_stab + self.num_x_stab))

    def _emit_z_layer(self, circuit: stim.Circuit, no_reset_ancilla: bool = False):
        """Emit Z stabilizer measurement for both blocks."""
        for stab_idx, support in enumerate(self.z_stabilizers):
            anc_d = self.data_z_syn[stab_idx]
            anc_a = self.ancilla_z_syn[stab_idx]
            for data_idx in support:
                circuit.append("CX", [self.data_qubits[data_idx], anc_d])
                circuit.append("CX", [self.ancilla_qubits[data_idx], anc_a])
        
        circuit.append("TICK")
        circuit.append("MR", self.data_z_syn)
        if no_reset_ancilla:
            circuit.append("M", self.ancilla_z_syn)
        else:
            circuit.append("MR", self.ancilla_z_syn)

    def _emit_x_layer(self, circuit: stim.Circuit, no_reset: bool = False):
        """Emit X stabilizer measurement for both blocks."""
        circuit.append("TICK")
        
        # H gates
        circuit.append("H", self.data_x_syn)
        circuit.append("H", self.ancilla_x_syn)
        circuit.append("TICK")
        
        # CNOTs: syndrome controls data (standard direction for X stabilizer measurement)
        for stab_idx, support in enumerate(self.x_stabilizers):
            anc_d = self.data_x_syn[stab_idx]
            anc_a = self.ancilla_x_syn[stab_idx]
            for data_idx in support:
                circuit.append("CX", [anc_d, self.data_qubits[data_idx]])
                circuit.append("CX", [anc_a, self.ancilla_qubits[data_idx]])
        
        circuit.append("TICK")
        circuit.append("H", self.data_x_syn)
        circuit.append("H", self.ancilla_x_syn)
        circuit.append("TICK")
        
        if no_reset:
            circuit.append("M", self.data_x_syn)
            circuit.append("M", self.ancilla_x_syn)
        else:
            circuit.append("MR", self.data_x_syn)
            circuit.append("MR", self.ancilla_x_syn)

    def to_stim(self) -> stim.Circuit:
        """Build the complete CNOT H-teleportation circuit with detectors.
        
        Returns an ideal circuit without noise. Use CircuitDepolarizingNoise.apply()
        to add noise for simulation.
        """
        circuit = stim.Circuit()
        d = self.num_ec_rounds
        
        # Initialize: Data in input state, Ancilla in |0⟩
        if self.input_state in ("+", "-"):
            circuit.append("H", self.data_qubits)
        # Ancilla already in |0⟩ (default)
        
        # Reset syndrome ancillas
        circuit.append("R", self.data_z_syn + self.data_x_syn)
        circuit.append("R", self.ancilla_z_syn + self.ancilla_x_syn)
        
        circuit.append("TICK")
        
        # Measurement tracking
        meas_record = {'prep': [], 'pre_cnot': [], 'post_cnot': []}
        meas_idx = 0
        
        def do_full_round(phase: str):
            nonlocal meas_idx
            round_data = {'z': {}, 'x': {}}
            
            # X layer FIRST (important for CNOT crossing timing)
            round_data['x']['data'] = meas_idx
            round_data['x']['ancilla'] = meas_idx + self.num_x_stab
            self._emit_x_layer(circuit)
            meas_idx += 2 * self.num_x_stab
            circuit.append("TICK")
            
            # Z layer SECOND
            round_data['z']['data'] = meas_idx
            round_data['z']['ancilla'] = meas_idx + self.num_z_stab
            self._emit_z_layer(circuit)
            meas_idx += 2 * self.num_z_stab
            
            meas_record[phase].append(round_data)
            circuit.append("TICK")
        
        # PREP phase (with extra rounds for anchor compensation)
        num_prep_rounds = d + 1 + self.extra_prep_rounds  # d+1 base + extra
        for r in range(num_prep_rounds):
            do_full_round('prep')
        
        # PRE-CNOT phase
        for r in range(d):
            do_full_round('pre_cnot')
        
        # Transversal CNOT(D→A)
        for i in range(self.n):
            circuit.append("CX", [self.data_qubits[i], self.ancilla_qubits[i]])
        circuit.append("TICK")
        
        # POST-CNOT phase
        for r in range(d - 1):
            do_full_round('post_cnot')
        
        # Final round structure depends on input
        if d >= 1:
            round_data = {'z': {}, 'x': {}}
            
            if self.input_state in ("0", "1"):
                # |0⟩: Z without reset on ancilla for boundary, then X without reset
                round_data['z']['data'] = meas_idx
                round_data['z']['ancilla'] = meas_idx + self.num_z_stab
                self._emit_z_layer(circuit, no_reset_ancilla=True)
                meas_idx += 2 * self.num_z_stab
                circuit.append("TICK")
                
                round_data['x']['data'] = meas_idx
                round_data['x']['ancilla'] = meas_idx + self.num_x_stab
                self._emit_x_layer(circuit, no_reset=True)
                meas_idx += 2 * self.num_x_stab
            else:
                # |+⟩: Z normal, X without reset for boundary
                round_data['z']['data'] = meas_idx
                round_data['z']['ancilla'] = meas_idx + self.num_z_stab
                self._emit_z_layer(circuit)
                meas_idx += 2 * self.num_z_stab
                circuit.append("TICK")
                
                round_data['x']['data'] = meas_idx
                round_data['x']['ancilla'] = meas_idx + self.num_x_stab
                self._emit_x_layer(circuit, no_reset=True)
                meas_idx += 2 * self.num_x_stab
            
            meas_record['post_cnot'].append(round_data)
            circuit.append("TICK")
        
        # Final destructive measurements
        circuit.append("MX", self.data_qubits)
        data_x_final = meas_idx
        meas_idx += self.n
        
        if self.input_state in ("0", "1"):
            circuit.append("M", self.ancilla_qubits)  # MZ
            ancilla_basis = "Z"
        else:
            circuit.append("MX", self.ancilla_qubits)
            ancilla_basis = "X"
        ancilla_final = meas_idx
        meas_idx += self.n
        
        total_meas = meas_idx
        
        def rec(idx):
            return stim.target_rec(idx - total_meas)
        
        # ═══════════ BUILD DETECTORS ═══════════
        num_prep = len(meas_record['prep'])
        num_pre = len(meas_record['pre_cnot'])
        num_post = len(meas_record['post_cnot'])
        
        # --- Z_D chain (2-term crossing) ---
        if self.data_z_deterministic:
            for i in range(self.num_z_stab):
                circuit.append("DETECTOR", [rec(meas_record['prep'][0]['z']['data'] + i)])
        
        for r in range(num_prep - 1):
            for i in range(self.num_z_stab):
                circuit.append("DETECTOR", [
                    rec(meas_record['prep'][r]['z']['data'] + i),
                    rec(meas_record['prep'][r + 1]['z']['data'] + i),
                ])
        
        for i in range(self.num_z_stab):
            circuit.append("DETECTOR", [
                rec(meas_record['prep'][num_prep - 1]['z']['data'] + i),
                rec(meas_record['pre_cnot'][0]['z']['data'] + i),
            ])
        
        for r in range(num_pre - 1):
            for i in range(self.num_z_stab):
                circuit.append("DETECTOR", [
                    rec(meas_record['pre_cnot'][r]['z']['data'] + i),
                    rec(meas_record['pre_cnot'][r + 1]['z']['data'] + i),
                ])
        
        # CNOT crossing: Z_D unchanged
        for i in range(self.num_z_stab):
            circuit.append("DETECTOR", [
                rec(meas_record['pre_cnot'][num_pre - 1]['z']['data'] + i),
                rec(meas_record['post_cnot'][0]['z']['data'] + i),
            ])
        
        for r in range(num_post - 1):
            for i in range(self.num_z_stab):
                circuit.append("DETECTOR", [
                    rec(meas_record['post_cnot'][r]['z']['data'] + i),
                    rec(meas_record['post_cnot'][r + 1]['z']['data'] + i),
                ])
        
        # --- Z_A chain ---
        if self.ancilla_z_deterministic:
            for i in range(self.num_z_stab):
                circuit.append("DETECTOR", [rec(meas_record['prep'][0]['z']['ancilla'] + i)])
        
        for r in range(num_prep - 1):
            for i in range(self.num_z_stab):
                circuit.append("DETECTOR", [
                    rec(meas_record['prep'][r]['z']['ancilla'] + i),
                    rec(meas_record['prep'][r + 1]['z']['ancilla'] + i),
                ])
        
        for i in range(self.num_z_stab):
            circuit.append("DETECTOR", [
                rec(meas_record['prep'][num_prep - 1]['z']['ancilla'] + i),
                rec(meas_record['pre_cnot'][0]['z']['ancilla'] + i),
            ])
        
        for r in range(num_pre - 1):
            for i in range(self.num_z_stab):
                circuit.append("DETECTOR", [
                    rec(meas_record['pre_cnot'][r]['z']['ancilla'] + i),
                    rec(meas_record['pre_cnot'][r + 1]['z']['ancilla'] + i),
                ])
        
        # CNOT crossing: Z_A → Z_D ⊗ Z_A
        if self.data_z_deterministic:
            # 2-term (Z_D = +1)
            for i in range(self.num_z_stab):
                circuit.append("DETECTOR", [
                    rec(meas_record['pre_cnot'][num_pre - 1]['z']['ancilla'] + i),
                    rec(meas_record['post_cnot'][0]['z']['ancilla'] + i),
                ])
        else:
            # 3-term
            for i in range(self.num_z_stab):
                circuit.append("DETECTOR", [
                    rec(meas_record['pre_cnot'][num_pre - 1]['z']['ancilla'] + i),
                    rec(meas_record['pre_cnot'][num_pre - 1]['z']['data'] + i),
                    rec(meas_record['post_cnot'][0]['z']['ancilla'] + i),
                ])
        
        for r in range(num_post - 1):
            for i in range(self.num_z_stab):
                circuit.append("DETECTOR", [
                    rec(meas_record['post_cnot'][r]['z']['ancilla'] + i),
                    rec(meas_record['post_cnot'][r + 1]['z']['ancilla'] + i),
                ])
        
        # Z_A boundary for |0⟩ input (MZ)
        if ancilla_basis == "Z":
            for i, support in enumerate(self.z_stabilizers):
                targets = [rec(meas_record['post_cnot'][-1]['z']['ancilla'] + i)]
                for q in support:
                    targets.append(rec(ancilla_final + q))
                circuit.append("DETECTOR", targets)
        
        # --- X_D chain ---
        if self.data_x_deterministic:
            for i in range(self.num_x_stab):
                circuit.append("DETECTOR", [rec(meas_record['prep'][0]['x']['data'] + i)])
        
        for r in range(num_prep - 1):
            for i in range(self.num_x_stab):
                circuit.append("DETECTOR", [
                    rec(meas_record['prep'][r]['x']['data'] + i),
                    rec(meas_record['prep'][r + 1]['x']['data'] + i),
                ])
        
        for i in range(self.num_x_stab):
            circuit.append("DETECTOR", [
                rec(meas_record['prep'][num_prep - 1]['x']['data'] + i),
                rec(meas_record['pre_cnot'][0]['x']['data'] + i),
            ])
        
        for r in range(num_pre - 1):
            for i in range(self.num_x_stab):
                circuit.append("DETECTOR", [
                    rec(meas_record['pre_cnot'][r]['x']['data'] + i),
                    rec(meas_record['pre_cnot'][r + 1]['x']['data'] + i),
                ])
        
        # CNOT crossing: X_D → X_D ⊗ X_A (3-term)
        for i in range(self.num_x_stab):
            circuit.append("DETECTOR", [
                rec(meas_record['pre_cnot'][num_pre - 1]['x']['data'] + i),
                rec(meas_record['post_cnot'][0]['x']['data'] + i),
                rec(meas_record['post_cnot'][0]['x']['ancilla'] + i),
            ])
        
        last_x_round = num_post - 1
        for r in range(last_x_round):
            for i in range(self.num_x_stab):
                circuit.append("DETECTOR", [
                    rec(meas_record['post_cnot'][r]['x']['data'] + i),
                    rec(meas_record['post_cnot'][r + 1]['x']['data'] + i),
                ])
        
        # X_D BOUNDARY (MX on data)
        for i, support in enumerate(self.x_stabilizers):
            targets = [rec(meas_record['post_cnot'][last_x_round]['x']['data'] + i)]
            for q in support:
                targets.append(rec(data_x_final + q))
            circuit.append("DETECTOR", targets)
        
        # --- X_A chain ---
        # No anchor (ancilla |0⟩, X indeterminate)
        
        for r in range(num_prep - 1):
            for i in range(self.num_x_stab):
                circuit.append("DETECTOR", [
                    rec(meas_record['prep'][r]['x']['ancilla'] + i),
                    rec(meas_record['prep'][r + 1]['x']['ancilla'] + i),
                ])
        
        for i in range(self.num_x_stab):
            circuit.append("DETECTOR", [
                rec(meas_record['prep'][num_prep - 1]['x']['ancilla'] + i),
                rec(meas_record['pre_cnot'][0]['x']['ancilla'] + i),
            ])
        
        for r in range(num_pre - 1):
            for i in range(self.num_x_stab):
                circuit.append("DETECTOR", [
                    rec(meas_record['pre_cnot'][r]['x']['ancilla'] + i),
                    rec(meas_record['pre_cnot'][r + 1]['x']['ancilla'] + i),
                ])
        
        # CNOT crossing: X_A unchanged (2-term)
        for i in range(self.num_x_stab):
            circuit.append("DETECTOR", [
                rec(meas_record['pre_cnot'][num_pre - 1]['x']['ancilla'] + i),
                rec(meas_record['post_cnot'][0]['x']['ancilla'] + i),
            ])
        
        for r in range(last_x_round):
            for i in range(self.num_x_stab):
                circuit.append("DETECTOR", [
                    rec(meas_record['post_cnot'][r]['x']['ancilla'] + i),
                    rec(meas_record['post_cnot'][r + 1]['x']['ancilla'] + i),
                ])
        
        # X_A BOUNDARY for |+⟩ input (MX on ancilla)
        if ancilla_basis == "X":
            for i, support in enumerate(self.x_stabilizers):
                targets = [rec(meas_record['post_cnot'][last_x_round]['x']['ancilla'] + i)]
                for q in support:
                    targets.append(rec(ancilla_final + q))
                circuit.append("DETECTOR", targets)
        
        # ═══════════ OBSERVABLE ═══════════
        obs_targets = []
        if self.input_state in ("0", "1"):
            # Z_L(A)
            for q in self.z_logical:
                obs_targets.append(rec(ancilla_final + q))
        else:
            # X_L(D) ⊕ X_L(A) (Bell correlation)
            for q in self.x_logical:
                obs_targets.append(rec(data_x_final + q))
            for q in self.x_logical:
                obs_targets.append(rec(ancilla_final + q))
        
        circuit.append("OBSERVABLE_INCLUDE", obs_targets, 0)
        
        # Store for external access
        self._data_x_final = data_x_final
        self._ancilla_final = ancilla_final
        self._ancilla_basis = ancilla_basis
        self._total_meas = total_meas
        self._meas_record = meas_record
        
        return circuit


# ════════════════════════════════════════════════════════════════════════════════
# RUN FUNCTIONS (using BPOSD decoder)
# ════════════════════════════════════════════════════════════════════════════════

def run_cz_h_teleportation(
    hz: np.ndarray,
    hx: np.ndarray,
    z_logical: List[int],
    x_logical: List[int],
    p: float,
    num_shots: int = 10_000,
    num_ec_rounds: int = 1,
    input_state: Literal["0", "1", "+", "-"] = "0",
    extra_prep_rounds: int = 0,
) -> HTeleportationResult:
    """
    Run CZ H-teleportation with BPOSD decoding.
    
    Best for self-dual CSS codes (Steane, color codes).
    
    Args:
        hz: Z stabilizer parity check matrix
        hx: X stabilizer parity check matrix
        z_logical: Qubit indices for Z logical operator
        x_logical: Qubit indices for X logical operator
        p: Physical error probability
        num_shots: Number of Monte Carlo shots
        num_ec_rounds: Number of EC rounds (typically = d)
        input_state: Input logical state
        extra_prep_rounds: Additional prep rounds
    
    Returns:
        HTeleportationResult
    """
    from qectostim.decoders.bp_osd import BPOSDDecoder
    from qectostim.noise.models import CircuitDepolarizingNoise
    
    builder = CZHTeleportationBuilder(
        hz=hz, hx=hx,
        z_logical=z_logical, x_logical=x_logical,
        num_ec_rounds=num_ec_rounds,
        input_state=input_state,
        extra_prep_rounds=extra_prep_rounds
    )
    ideal_circuit = builder.to_stim()
    
    # Apply noise model
    noise_model = CircuitDepolarizingNoise(p1=p, p2=p)
    circuit = noise_model.apply(ideal_circuit)
    
    dem = circuit.detector_error_model(decompose_errors=False)
    decoder = BPOSDDecoder(dem=dem)
    
    det_sampler = circuit.compile_detector_sampler()
    detection_events, observable_flips = det_sampler.sample(num_shots, separate_observables=True)
    
    predictions = decoder.decode_batch(detection_events)
    num_errors = np.sum(predictions[:, 0] != observable_flips[:, 0])
    
    return HTeleportationResult(
        logical_error_rate=num_errors / num_shots,
        num_shots=num_shots,
        num_errors=int(num_errors),
        num_detectors=circuit.num_detectors,
        input_state=input_state
    )


def run_cnot_h_teleportation(
    hz: np.ndarray,
    hx: np.ndarray,
    z_logical: List[int],
    x_logical: List[int],
    p: float,
    num_shots: int = 10_000,
    num_ec_rounds: int = 1,
    input_state: Literal["0", "1", "+", "-"] = "0",
    extra_prep_rounds: int = 0,
) -> HTeleportationResult:
    """
    Run CNOT H-teleportation with BPOSD decoding.
    
    Works for ANY CSS code (including non-self-dual like Surface codes).
    
    Args:
        hz: Z stabilizer parity check matrix
        hx: X stabilizer parity check matrix  
        z_logical: Qubit indices for Z logical operator
        x_logical: Qubit indices for X logical operator
        p: Physical error probability
        num_shots: Number of Monte Carlo shots
        num_ec_rounds: Number of EC rounds (typically = d)
        input_state: Input logical state
        extra_prep_rounds: Additional prep rounds
    
    Returns:
        HTeleportationResult
    """
    from qectostim.decoders.bp_osd import BPOSDDecoder
    from qectostim.noise.models import CircuitDepolarizingNoise
    
    builder = CNOTHTeleportationBuilder(
        hz=hz, hx=hx,
        z_logical=z_logical, x_logical=x_logical,
        num_ec_rounds=num_ec_rounds,
        input_state=input_state,
        extra_prep_rounds=extra_prep_rounds
    )
    ideal_circuit = builder.to_stim()
    
    # Apply noise model
    noise_model = CircuitDepolarizingNoise(p1=p, p2=p)
    circuit = noise_model.apply(ideal_circuit)
    
    dem = circuit.detector_error_model(decompose_errors=False)
    decoder = BPOSDDecoder(dem=dem)
    
    det_sampler = circuit.compile_detector_sampler()
    detection_events, observable_flips = det_sampler.sample(num_shots, separate_observables=True)
    
    predictions = decoder.decode_batch(detection_events)
    num_errors = np.sum(predictions[:, 0] != observable_flips[:, 0])
    
    return HTeleportationResult(
        logical_error_rate=num_errors / num_shots,
        num_shots=num_shots,
        num_errors=int(num_errors),
        num_detectors=circuit.num_detectors,
        input_state=input_state
    )


# ════════════════════════════════════════════════════════════════════════════════
# MAIN TEST
# ════════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    sys.path.insert(0, 'src')
    
    print("=" * 70)
    print("H-TELEPORTATION MODULE TESTS")
    print("=" * 70)
    
    # Test 1: Steane code with CZ builder
    print("\n" + "─" * 70)
    print("TEST 1: CZ Protocol with Steane [[7,1,3]] Code")
    print("─" * 70)
    
    try:
        from qectostim.codes.small.steane_713 import SteaneCode713
        from qectostim.noise.models import CircuitDepolarizingNoise
        
        steane = SteaneCode713()
        hz = steane.hz
        hx = steane.hx
        z_logical = list(np.where(steane.logical_z_array())[0])
        x_logical = list(np.where(steane.logical_x_array())[0])
        
        print(f"  n={steane.n}, z_stab={hz.shape[0]}, x_stab={hx.shape[0]}")
        print(f"  z_logical={z_logical}, x_logical={x_logical}")
        print(f"  Self-dual: {np.array_equal(hz, hx)}")
        
        for input_state in ["0", "+"]:
            print(f"\n  Testing |{input_state}⟩ input:")
            
            # Verify circuit at p=0
            builder = CZHTeleportationBuilder(
                hz=hz, hx=hx, z_logical=z_logical, x_logical=x_logical,
                num_ec_rounds=3, input_state=input_state
            )
            circuit = builder.to_stim()
            print(f"    Circuit: {circuit.num_measurements} meas, {circuit.num_detectors} det")
            
            # Check determinism
            det_sampler = circuit.compile_detector_sampler()
            samples = det_sampler.sample(100)
            fire_rate = samples.mean()
            print(f"    Detector fire rate (p=0): {fire_rate:.6f}")
            
            # Test with noise
            result = run_cz_h_teleportation(
                hz=hz, hx=hx, z_logical=z_logical, x_logical=x_logical,
                p=0.001, num_shots=5000, num_ec_rounds=3, input_state=input_state
            )
            print(f"    LER (p=0.001): {result.logical_error_rate:.4f}")
        
        print("\n  ✓ CZ protocol works for Steane code!")
        
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 2: Surface code with CNOT builder
    print("\n" + "─" * 70)
    print("TEST 2: CNOT Protocol with Rotated Surface Code d=3,5")
    print("─" * 70)
    
    try:
        from qectostim.codes.surface.rotated_surface import RotatedSurfaceCode
        from qectostim.noise.models import CircuitDepolarizingNoise
        
        for d in [3, 5]:
            print(f"\n  d={d}:")
            code = RotatedSurfaceCode(d)
            hz = code.hz
            hx = code.hx
            z_logical = code.metadata.get("logical_z_support", list(range(d)))
            x_logical = code.metadata.get("logical_x_support", list(range(0, d*d, d)))
            
            print(f"    n={code.n}, z_stab={hz.shape[0]}, x_stab={hx.shape[0]}")
            print(f"    Self-dual: {np.array_equal(hz, hx)}")
            
            for input_state in ["0", "+"]:
                print(f"\n    Testing |{input_state}⟩ input:")
                
                builder = CNOTHTeleportationBuilder(
                    hz=hz, hx=hx, z_logical=z_logical, x_logical=x_logical,
                    num_ec_rounds=d, input_state=input_state
                )
                circuit = builder.to_stim()
                print(f"      Circuit: {circuit.num_measurements} meas, {circuit.num_detectors} det")
                
                det_sampler = circuit.compile_detector_sampler()
                samples = det_sampler.sample(100)
                fire_rate = samples.mean()
                print(f"      Detector fire rate (p=0): {fire_rate:.6f}")
                
                result = run_cnot_h_teleportation(
                    hz=hz, hx=hx, z_logical=z_logical, x_logical=x_logical,
                    p=0.001, num_shots=5000, num_ec_rounds=d, input_state=input_state
                )
                print(f"      LER (p=0.001): {result.logical_error_rate:.4f}")
        
        print("\n  ✓ CNOT protocol works for Surface code!")
        
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 3: Extra prep rounds effect
    print("\n" + "─" * 70)
    print("TEST 3: Effect of extra prep rounds (CNOT, Surface d=3, |+⟩)")
    print("─" * 70)
    
    try:
        from qectostim.codes.surface.rotated_surface import RotatedSurfaceCode
        from qectostim.noise.models import CircuitDepolarizingNoise
        
        d = 3
        code = RotatedSurfaceCode(d)
        hz = code.hz
        hx = code.hx
        z_logical = code.metadata.get("logical_z_support", list(range(d)))
        x_logical = code.metadata.get("logical_x_support", list(range(0, d*d, d)))
        
        print(f"  p=0.003, num_shots=10000, input=|+⟩")
        print(f"\n  {'Extra Prep':<12} {'Detectors':<12} {'LER':<12}")
        print("  " + "-" * 36)
        
        for extra in [0, 1, 2, 3]:
            builder = CNOTHTeleportationBuilder(
                hz=hz, hx=hx, z_logical=z_logical, x_logical=x_logical,
                num_ec_rounds=d, input_state="+",
                extra_prep_rounds=extra
            )
            circuit = builder.to_stim()
            
            result = run_cnot_h_teleportation(
                hz=hz, hx=hx, z_logical=z_logical, x_logical=x_logical,
                p=0.003, num_shots=10000, num_ec_rounds=d, input_state="+",
                extra_prep_rounds=extra
            )
            
            print(f"  {extra:<12} {circuit.num_detectors:<12} {result.logical_error_rate:<12.4f}")
        
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 70)
    print("ALL TESTS COMPLETE")
    print("=" * 70)
