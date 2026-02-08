"""
Logical state preparation via stabilizer projection.

This module implements fault-tolerant state preparation protocols for CSS codes.

═══════════════════════════════════════════════════════════════════════════════
FUNDAMENTAL PROTOCOL: CSS STATE PREPARATION BY PROJECTION
═══════════════════════════════════════════════════════════════════════════════

For CSS codes, state preparation is simple:

    |0⟩_L: Initialize ALL physical qubits to |0⟩, then measure stabilizers
    |+⟩_L: Initialize ALL physical qubits to |+⟩, then measure stabilizers

WHY THIS WORKS:
- |0⟩^⊗n is stabilized by all Z_i single-qubit Paulis
- Measuring X stabilizers projects into their joint eigenspace
- Since all Z stabilizers commute with |0⟩^⊗n, they're already satisfied
- The measurement outcomes define the Pauli frame (don't force to +1!)

PAULI FRAME PHILOSOPHY:
- DON'T try to make all stabilizers be +1 eigenvalues
- Instead, RECORD the initial stabilizer measurements as the reference
- When checking stabilizers later, COMPARE to these initial values
- This eliminates complex encoding circuits and Gaussian elimination

═══════════════════════════════════════════════════════════════════════════════
ADVANCED PROTOCOL: SINGLE-QUBIT GATE INJECTION
═══════════════════════════════════════════════════════════════════════════════

To prepare U|0⟩_L for gates like H, S, T:
1. Initialize all physical qubits to |0⟩
2. Apply physical U to ONE qubit at the intersection of X_L and Z_L supports
3. Measure all stabilizers (defines Pauli frame)
4. Result: U|0⟩_L up to the tracked Pauli frame

This works because the single-qubit state "spreads" into the logical state
via the stabilizer projection.

═══════════════════════════════════════════════════════════════════════════════
CORRECT STABILIZER MEASUREMENT CIRCUITS
═══════════════════════════════════════════════════════════════════════════════

Z stabilizer (measures ∏Z_i):
    ancilla: R ─────●─────●─────M   (NO Hadamards!)
                    │     │
    data[i]: ──────⊕─────┼─────
    data[j]: ────────────⊕─────
    
    Circuit: R - CX[data→anc] for each qubit in support - M

X stabilizer (measures ∏X_i):
    ancilla: R ── H ──●────●── H ── M
                      │    │
    data[i]: ────────⊕────┼────────
    data[j]: ─────────────⊕────────
    
    Circuit: R - H - CX[anc→data] for each qubit in support - H - M

═══════════════════════════════════════════════════════════════════════════════
FAULT-TOLERANT PREPARATION
═══════════════════════════════════════════════════════════════════════════════

For a distance-d code, use d rounds of stabilizer measurements.
- Consecutive rounds are compared via DETECTORS
- Measurement errors manifest as detector firings
- Up to floor((d-1)/2) measurement errors can be corrected

═══════════════════════════════════════════════════════════════════════════════
References:
- Pauli frame tracking (Knill 2005, Raussendorf 2007)
- CSS code preparation (Steane 1996, Calderbank-Shor 1996)
- Gate teleportation (Gottesman-Chuang 1999)
═══════════════════════════════════════════════════════════════════════════════
"""

from typing import List, Optional, Tuple, Dict, Any, Set
from dataclasses import dataclass
from enum import Enum, auto
import numpy as np
import stim


# ═══════════════════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════════════════════

class LogicalBasis(Enum):
    """Basis for logical state preparation."""
    Z = auto()  # |0⟩_L or |1⟩_L
    X = auto()  # |+⟩_L or |-⟩_L


class InjectionGate(Enum):
    """Physical gates that can be injected for teleportation."""
    IDENTITY = auto()  # Prepare |0⟩_L (no gate)
    H = auto()         # Prepare H|0⟩_L = |+⟩_L
    S = auto()         # Prepare S|0⟩_L = |0⟩_L (S has no effect on |0⟩)
    S_DAG = auto()     # Prepare S†|0⟩_L
    T = auto()         # Prepare T|0⟩_L (magic state, non-Clifford)
    T_DAG = auto()     # Prepare T†|0⟩_L


@dataclass
class ProjectionResult:
    """Result of CSS state preparation by projection."""
    x_stab_meas: List[List[int]]  # X stabilizer measurements per round (local indices)
    z_stab_meas: List[List[int]]  # Z stabilizer measurements per round (local indices)
    num_rounds: int
    basis: LogicalBasis
    meas_base: int = 0  # Global measurement index offset for conversion
    
    @property
    def final_x_meas(self) -> List[int]:
        """Final round X stabilizer measurements (local indices)."""
        return self.x_stab_meas[-1] if self.x_stab_meas else []
    
    @property
    def final_z_meas(self) -> List[int]:
        """Final round Z stabilizer measurements (local indices)."""
        return self.z_stab_meas[-1] if self.z_stab_meas else []
    
    @property
    def final_x_meas_global(self) -> List[int]:
        """Final round X stabilizer measurements as global indices."""
        return [self.meas_base + i for i in self.final_x_meas]
    
    @property
    def final_z_meas_global(self) -> List[int]:
        """Final round Z stabilizer measurements as global indices."""
        return [self.meas_base + i for i in self.final_z_meas]
    
    def to_global(self, local_idx: int) -> int:
        """Convert local measurement index to global."""
        return self.meas_base + local_idx
    
    @property
    def all_measurements(self) -> List[int]:
        """All projection measurements (flattened, local indices)."""
        meas = []
        for round_x, round_z in zip(self.x_stab_meas, self.z_stab_meas):
            meas.extend(round_z)
            meas.extend(round_x)
        return meas
    
    @property
    def all_measurements_global(self) -> List[int]:
        """All projection measurements as global indices."""
        return [self.meas_base + i for i in self.all_measurements]


@dataclass
class InjectionResult:
    """Result of logical state injection via projection."""
    injection_qubit_local: int   # Local index within the code
    injection_qubit_global: int  # Global qubit index
    x_stab_meas: List[int]       # Final round X stabilizer measurement indices
    z_stab_meas: List[int]       # Final round Z stabilizer measurement indices
    all_projection_meas: List[int]  # All projection measurements
    num_rounds: int              # Number of projection rounds used


@dataclass 
class ProjectionRoundResult:
    """Result of one round of stabilizer projection."""
    x_meas_indices: List[int]    # X stabilizer measurement indices this round
    z_meas_indices: List[int]    # Z stabilizer measurement indices this round


# ═══════════════════════════════════════════════════════════════════════════════
# CSS STATE PREPARATION BY PROJECTION
# ═══════════════════════════════════════════════════════════════════════════════

class CSSStatePreparation:
    """
    Prepare logical states for CSS codes by projection.
    
    This is the fundamental, simple protocol:
    - |0⟩_L: Initialize |0⟩^⊗n, measure stabilizers
    - |+⟩_L: Initialize |+⟩^⊗n, measure stabilizers
    
    The stabilizer measurement outcomes define the Pauli frame.
    Don't try to force them to +1!
    
    Parameters
    ----------
    code : CSSCode
        The CSS code (must have hx and hz attributes).
        
    Example
    -------
    >>> from qectostim.codes.small.steane_713 import SteaneCode713
    >>> code = SteaneCode713()
    >>> prep = CSSStatePreparation(code)
    >>> 
    >>> circuit = stim.Circuit()
    >>> data_qubits = list(range(7))
    >>> ancilla = 7
    >>> 
    >>> # Prepare |0⟩_L
    >>> result = prep.prepare_zero(circuit, data_qubits, ancilla, num_rounds=1)
    >>> 
    >>> # The result.final_x_meas and result.final_z_meas define the Pauli frame
    """
    
    def __init__(self, code):
        """
        Initialize for a given CSS code.
        
        Parameters
        ----------
        code : CSSCode
            Must have `hx` (X stabilizer matrix) and `hz` (Z stabilizer matrix).
        """
        self.code = code
        self.hx = code.hx
        self.hz = code.hz
        self.n = code.n
        self.n_x = self.hx.shape[0] if self.hx is not None else 0
        self.n_z = self.hz.shape[0] if self.hz is not None else 0
    
    def prepare_zero(
        self,
        circuit: stim.Circuit,
        data_qubits: List[int],
        ancilla: int,
        num_rounds: int = 1,
        emit_detectors: bool = True,
        noise_model=None,
    ) -> ProjectionResult:
        """
        Prepare |0⟩_L by initializing |0⟩^⊗n and measuring stabilizers.
        
        Protocol:
        1. Reset all data qubits (puts them in |0⟩)
        2. Measure X stabilizers (projects into X stabilizer eigenspace)
        3. Measure Z stabilizers (already satisfied, but needed for frame)
        
        Parameters
        ----------
        circuit : stim.Circuit
            Circuit to append to.
        data_qubits : List[int]
            Global indices of the n data qubits.
        ancilla : int
            Global index of the ancilla qubit for measurements.
        num_rounds : int
            Number of stabilizer measurement rounds. Use d rounds for
            fault-tolerant preparation of a distance-d code.
        emit_detectors : bool
            Whether to emit detectors comparing consecutive rounds.
        noise_model : NoiseModel, optional
            Noise model for adding errors to operations.
            
        Returns
        -------
        ProjectionResult
            Contains measurement indices for Pauli frame tracking.
        """
        # Reset all data qubits to |0⟩
        circuit.append("R", data_qubits)
        circuit.append("TICK")
        
        return self._measure_stabilizers(
            circuit, data_qubits, ancilla, num_rounds,
            emit_detectors, LogicalBasis.Z, noise_model
        )
    
    def prepare_plus(
        self,
        circuit: stim.Circuit,
        data_qubits: List[int],
        ancilla: int,
        num_rounds: int = 1,
        emit_detectors: bool = True,
        noise_model=None,
    ) -> ProjectionResult:
        """
        Prepare |+⟩_L by initializing |+⟩^⊗n and measuring stabilizers.
        
        Protocol:
        1. Reset all data qubits, then apply H to each (puts them in |+⟩)
        2. Measure Z stabilizers (projects into Z stabilizer eigenspace)
        3. Measure X stabilizers (already satisfied, but needed for frame)
        
        Parameters
        ----------
        circuit : stim.Circuit
            Circuit to append to.
        data_qubits : List[int]
            Global indices of the n data qubits.
        ancilla : int
            Global index of the ancilla qubit for measurements.
        num_rounds : int
            Number of stabilizer measurement rounds.
        emit_detectors : bool
            Whether to emit detectors comparing consecutive rounds.
        noise_model : NoiseModel, optional
            Noise model for adding errors to operations.
            
        Returns
        -------
        ProjectionResult
            Contains measurement indices for Pauli frame tracking.
        """
        # Reset all data qubits to |0⟩, then H to get |+⟩
        circuit.append("R", data_qubits)
        circuit.append("H", data_qubits)
        circuit.append("TICK")
        
        return self._measure_stabilizers(
            circuit, data_qubits, ancilla, num_rounds,
            emit_detectors, LogicalBasis.X, noise_model
        )
    
    def _measure_stabilizers(
        self,
        circuit: stim.Circuit,
        data_qubits: List[int],
        ancilla: int,
        num_rounds: int,
        emit_detectors: bool,
        basis: LogicalBasis,
        noise_model=None,
    ) -> ProjectionResult:
        """
        Measure all stabilizers for num_rounds.
        
        Returns ProjectionResult with all measurement indices.
        """
        all_x_meas: List[List[int]] = []
        all_z_meas: List[List[int]] = []
        
        meas_counter = 0
        
        for round_idx in range(num_rounds):
            z_meas = []
            x_meas = []
            
            # ═══════════════════════════════════════════════════════════════
            # Z-TYPE STABILIZERS
            # Circuit: R - CX[data→anc] - M (NO Hadamards!)
            # ═══════════════════════════════════════════════════════════════
            if self.hz is not None:
                for stab_idx in range(self.n_z):
                    circuit.append("R", [ancilla])
                    support = list(np.where(self.hz[stab_idx])[0])
                    for q in support:
                        circuit.append("CX", [data_qubits[q], ancilla])
                    circuit.append("M", [ancilla])
                    z_meas.append(meas_counter)
                    meas_counter += 1
            
            # ═══════════════════════════════════════════════════════════════
            # X-TYPE STABILIZERS  
            # Circuit: R - H - CX[anc→data] - H - M
            # ═══════════════════════════════════════════════════════════════
            if self.hx is not None:
                for stab_idx in range(self.n_x):
                    circuit.append("R", [ancilla])
                    circuit.append("H", [ancilla])
                    support = list(np.where(self.hx[stab_idx])[0])
                    for q in support:
                        circuit.append("CX", [ancilla, data_qubits[q]])
                    circuit.append("H", [ancilla])
                    circuit.append("M", [ancilla])
                    x_meas.append(meas_counter)
                    meas_counter += 1
            
            circuit.append("TICK")
            all_z_meas.append(z_meas)
            all_x_meas.append(x_meas)
        
        # ═══════════════════════════════════════════════════════════════════
        # EMIT DETECTORS (compare consecutive rounds)
        # ═══════════════════════════════════════════════════════════════════
        if emit_detectors and num_rounds > 1:
            for round_idx in range(num_rounds - 1):
                # Z stabilizer detectors
                for stab_idx in range(self.n_z):
                    prev = all_z_meas[round_idx][stab_idx]
                    curr = all_z_meas[round_idx + 1][stab_idx]
                    circuit.append("DETECTOR", [
                        stim.target_rec(prev - meas_counter),
                        stim.target_rec(curr - meas_counter),
                    ])
                
                # X stabilizer detectors
                for stab_idx in range(self.n_x):
                    prev = all_x_meas[round_idx][stab_idx]
                    curr = all_x_meas[round_idx + 1][stab_idx]
                    circuit.append("DETECTOR", [
                        stim.target_rec(prev - meas_counter),
                        stim.target_rec(curr - meas_counter),
                    ])
        
        return ProjectionResult(
            x_stab_meas=all_x_meas,
            z_stab_meas=all_z_meas,
            num_rounds=num_rounds,
            basis=basis,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# PAULI FRAME UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════

def get_frame_measurements_for_observable(
    code,
    stab_meas: List[int],
    stab_matrix: np.ndarray,
    logical_op: np.ndarray,
) -> List[int]:
    """
    Find which stabilizer measurements contribute to a logical observable's frame.
    
    A stabilizer measurement contributes to the logical observable's Pauli frame
    if and only if the stabilizer has ODD overlap with the logical operator.
    
    Parameters
    ----------
    code : CSSCode
        The CSS code.
    stab_meas : List[int]
        Measurement indices for the stabilizers.
    stab_matrix : np.ndarray
        The stabilizer matrix (hx or hz).
    logical_op : np.ndarray
        The logical operator (row of Lx or Lz).
        
    Returns
    -------
    List[int]
        Measurement indices that must be XORed into the observable.
    """
    if stab_matrix is None or len(stab_meas) == 0:
        return []
    
    logical_op = np.atleast_1d(logical_op)
    l_support = set(np.where(logical_op)[0])
    frame_meas = []
    
    for stab_idx, meas_idx in enumerate(stab_meas):
        if stab_idx < stab_matrix.shape[0]:
            stab_support = set(np.where(stab_matrix[stab_idx])[0])
            overlap = len(stab_support & l_support)
            if overlap % 2 == 1:
                frame_meas.append(meas_idx)
    
    return frame_meas


def get_x_frame_for_z_measurement(
    code,
    prep_result: ProjectionResult,
) -> List[int]:
    """
    Get the Pauli frame measurements for measuring Z_L after preparing |0⟩_L.
    
    When we prepare |0⟩_L via projection and then measure Z_L, the result
    depends on X stabilizer measurements that have odd overlap with Z_L.
    
    Wait... actually for |0⟩_L prepared from |0⟩^⊗n:
    - Z_L has definite eigenvalue determined by which codeword we're in
    - The X stabilizer measurements don't flip Z_L (they commute!)
    
    The frame for Z_L comes from Z stabilizers with odd overlap... but
    Z stabilizers commute with Z_L, so no frame contribution.
    
    Actually: For |0⟩_L, Z_L is deterministic (+1). No frame needed.
    For |+⟩_L, Z_L is random (±1 with equal probability).
    """
    # For |0⟩_L prepared from |0⟩^⊗n, Z_L = +1 deterministically
    # No frame measurements needed
    return []


def get_z_frame_for_x_measurement(
    code,
    prep_result: ProjectionResult,
) -> List[int]:
    """
    Get the Pauli frame measurements for measuring X_L after preparing |+⟩_L.
    
    When we prepare |+⟩_L via projection and then measure X_L, the result
    depends on Z stabilizer measurements that have odd overlap with X_L.
    
    Similarly to above: For |+⟩_L, X_L = +1 deterministically. No frame needed.
    """
    return []


# ═══════════════════════════════════════════════════════════════════════════════
# SINGLE-QUBIT INJECTION (for gate teleportation)
# ═══════════════════════════════════════════════════════════════════════════════

def find_injection_qubit(code) -> int:
    """
    Find the qubit at the intersection of X_L and Z_L supports.
    
    For injection to produce U|0⟩_L correctly, the physical gate must be
    applied to a qubit in BOTH X_L and Z_L supports.
    
    Parameters
    ----------
    code : CSSCode
        The CSS code.
        
    Returns
    -------
    int
        Local qubit index for injection.
        
    Raises
    ------
    ValueError
        If no intersection exists.
    """
    # Get logical operators (requires CSSCode with Lx/Lz properties)
    css = code.as_css()
    if css is None:
        raise ValueError(f"Code {type(code).__name__} is not a CSS code; injection requires Lx/Lz")
    
    lx = np.atleast_2d(css.Lx)
    lz = np.atleast_2d(css.Lz)
    
    # Get supports for logical qubit 0
    x_support = set(np.where(lx[0])[0])
    z_support = set(np.where(lz[0])[0])
    
    intersection = x_support & z_support
    
    if not intersection:
        raise ValueError(
            f"No qubit in both X_L and Z_L supports.\n"
            f"X_L support: {sorted(x_support)}\n"
            f"Z_L support: {sorted(z_support)}"
        )
    
    return min(intersection)  # Deterministic choice


def get_injection_gates(gate: InjectionGate) -> List[str]:
    """Get Stim gate names for injection."""
    gate_map = {
        InjectionGate.IDENTITY: [],
        InjectionGate.H: ["H"],
        InjectionGate.S: ["S"],
        InjectionGate.S_DAG: ["S_DAG"],
        InjectionGate.T: [],  # Non-Clifford, can't simulate in Stim
        InjectionGate.T_DAG: [],
    }
    return gate_map.get(gate, [])


class LogicalStateInjector:
    """
    Prepares logical states via single-qubit injection and stabilizer projection.
    
    PROTOCOL:
    1. All qubits start in |0⟩
    2. Apply physical U to injection qubit (at X_L ∩ Z_L intersection)
    3. Measure stabilizers d times for FT preparation (outcomes define Pauli frame)
    4. Result: U|0⟩_L up to tracked Pauli
    
    For fault-tolerant preparation of a distance-d code, use d rounds
    of stabilizer measurements. This allows correction of up to
    floor((d-1)/2) measurement errors via majority voting.
    
    The final round's projection measurements MUST be included in 
    OBSERVABLE_INCLUDE for deterministic results.
    """
    
    def __init__(self, code, block_name: str = "injection_block"):
        """
        Initialize for a given code.
        
        Parameters
        ----------
        code : CSSCode
            The CSS code.
        block_name : str
            Name of the block being prepared.
        """
        self.code = code
        self.block_name = block_name
        self._injection_qubit = find_injection_qubit(code)
        
        # Get stabilizer matrices
        self.hx = code.hx
        self.hz = code.hz
        self.n_x = self.hx.shape[0] if self.hx is not None else 0
        self.n_z = self.hz.shape[0] if self.hz is not None else 0
        
    @property
    def injection_qubit(self) -> int:
        """Local index of the injection qubit."""
        return self._injection_qubit
    
    def _emit_one_projection_round(
        self,
        circuit: stim.Circuit,
        data_qubits: List[int],
        proj_ancilla: int,
        ctx: Optional[Any] = None,
        meas_counter: int = 0,
    ) -> Tuple[ProjectionRoundResult, int]:
        """
        Emit one round of stabilizer projection measurements.
        
        CORRECT CIRCUITS:
        - Z stabilizer: R-CX[data→anc]-M (data controls ancilla)
        - X stabilizer: R-H-CX[anc→data]-H-M (ancilla controls data)
        
        Returns
        -------
        Tuple[ProjectionRoundResult, int]
            The round result and updated measurement counter.
        """
        z_meas = []
        x_meas = []
        
        # === Z-type stabilizers ===
        # Circuit: R-CX[data→anc]-M (NO H gates!)
        if self.hz is not None and self.n_z > 0:
            for row_idx in range(self.n_z):
                circuit.append("R", [proj_ancilla])
                support = list(np.where(self.hz[row_idx])[0])
                for q in support:
                    circuit.append("CX", [data_qubits[q], proj_ancilla])
                
                if ctx is not None:
                    meas_idx = ctx.add_measurement(1)
                else:
                    meas_idx = meas_counter
                    meas_counter += 1
                circuit.append("M", [proj_ancilla])
                z_meas.append(meas_idx)
        
        # === X-type stabilizers ===
        # Circuit: R-H-CX[anc→data]-H-M
        if self.hx is not None and self.n_x > 0:
            for row_idx in range(self.n_x):
                circuit.append("R", [proj_ancilla])
                circuit.append("H", [proj_ancilla])
                support = list(np.where(self.hx[row_idx])[0])
                for q in support:
                    circuit.append("CX", [proj_ancilla, data_qubits[q]])
                circuit.append("H", [proj_ancilla])
                
                if ctx is not None:
                    meas_idx = ctx.add_measurement(1)
                else:
                    meas_idx = meas_counter
                    meas_counter += 1
                circuit.append("M", [proj_ancilla])
                x_meas.append(meas_idx)
        
        circuit.append("TICK")
        
        return ProjectionRoundResult(x_meas_indices=x_meas, z_meas_indices=z_meas), meas_counter
    
    def emit_injection_and_projection(
        self,
        circuit: stim.Circuit,
        data_qubits: List[int],
        proj_ancilla: int,
        gate: InjectionGate,
        ctx: Optional[Any] = None,
        num_rounds: int = 1,
        emit_detectors: bool = True,
    ) -> InjectionResult:
        """
        Emit the complete injection + projection protocol.
        
        Parameters
        ----------
        circuit : stim.Circuit
            Circuit to emit into.
        data_qubits : List[int]
            Global indices of data qubits for this block.
        proj_ancilla : int
            Global index of the projection ancilla qubit.
        gate : InjectionGate
            The gate to inject.
        ctx : DetectorContext, optional
            Context for measurement index tracking.
        num_rounds : int
            Number of projection rounds. Use d rounds for distance-d code
            for fault-tolerant preparation.
        emit_detectors : bool
            Whether to emit detectors comparing consecutive rounds.
            
        Returns
        -------
        InjectionResult
            Contains measurement indices for Pauli frame tracking.
        """
        injection_qubit_local = self._injection_qubit
        injection_qubit_global = data_qubits[injection_qubit_local]
        
        # =================================================================
        # STEP 1: INJECTION
        # Apply physical gate U to the injection qubit
        # =================================================================
        for gate_name in get_injection_gates(gate):
            circuit.append(gate_name, [injection_qubit_global])
        
        circuit.append("TICK")
        
        # =================================================================
        # STEP 2: STABILIZER PROJECTION (num_rounds times for FT)
        # =================================================================
        all_rounds: List[ProjectionRoundResult] = []
        meas_counter = 0
        
        for round_idx in range(num_rounds):
            round_result, meas_counter = self._emit_one_projection_round(
                circuit, data_qubits, proj_ancilla, ctx, meas_counter
            )
            all_rounds.append(round_result)
        
        # =================================================================
        # STEP 3: EMIT DETECTORS (compare consecutive rounds)
        # =================================================================
        if emit_detectors and num_rounds > 1:
            # Calculate total measurements for lookback indices
            total_meas = sum(
                len(r.x_meas_indices) + len(r.z_meas_indices) 
                for r in all_rounds
            )
            
            for round_idx in range(num_rounds - 1):
                prev = all_rounds[round_idx]
                curr = all_rounds[round_idx + 1]
                
                # Z stabilizer detectors
                for stab_idx in range(self.n_z):
                    prev_idx = prev.z_meas_indices[stab_idx]
                    curr_idx = curr.z_meas_indices[stab_idx]
                    circuit.append("DETECTOR", [
                        stim.target_rec(prev_idx - total_meas),
                        stim.target_rec(curr_idx - total_meas),
                    ])
                
                # X stabilizer detectors
                for stab_idx in range(self.n_x):
                    prev_idx = prev.x_meas_indices[stab_idx]
                    curr_idx = curr.x_meas_indices[stab_idx]
                    circuit.append("DETECTOR", [
                        stim.target_rec(prev_idx - total_meas),
                        stim.target_rec(curr_idx - total_meas),
                    ])
        
        # Final round measurements for Pauli frame
        final_round = all_rounds[-1]
        all_meas = []
        for r in all_rounds:
            all_meas.extend(r.z_meas_indices)
            all_meas.extend(r.x_meas_indices)
        
        return InjectionResult(
            injection_qubit_local=injection_qubit_local,
            injection_qubit_global=injection_qubit_global,
            x_stab_meas=final_round.x_meas_indices,
            z_stab_meas=final_round.z_meas_indices,
            all_projection_meas=all_meas,
            num_rounds=num_rounds,
        )
    
    def emit_injection_only(
        self,
        circuit: stim.Circuit,
        data_qubits: List[int],
        gate: InjectionGate,
    ) -> int:
        """
        Emit only the injection step (no projection).
        
        Use this when projecting with a different method or when
        the state is already in the codespace.
        
        Returns the global index of the injection qubit.
        """
        injection_qubit_global = data_qubits[self._injection_qubit]
        
        for gate_name in get_injection_gates(gate):
            circuit.append(gate_name, [injection_qubit_global])
        
        circuit.append("TICK")
        return injection_qubit_global


def get_pauli_frame_measurements_for_x_observable(
    code,
    x_stab_meas: List[int],
    z_stab_meas: List[int],
) -> List[int]:
    """
    Determine which projection measurements affect the X_L observable.
    
    For |+⟩_L prepared via injection, the X_L eigenvalue depends on
    the X stabilizer projection outcomes that overlap with X_L.
    
    The rule is:
    - X stabilizers that have ODD overlap with X_L contribute to the frame
    - Z stabilizers don't affect X eigenvalue (they commute with X_L)
    
    Parameters
    ----------
    code : CSSCode
        The CSS code.
    x_stab_meas : List[int]
        Measurement indices for X stabilizer projections.
    z_stab_meas : List[int]
        Measurement indices for Z stabilizer projections.
        
    Returns
    -------
    List[int]
        Measurement indices that must be XORed into X_L observable.
    """
    css = code.as_css()
    if css is None:
        return []
    
    lx = np.atleast_2d(css.Lx)
    hx = css.hx
    
    x_l_support = set(np.where(lx[0])[0])
    frame_meas = []
    
    # Check which X stabilizers have odd overlap with X_L
    for stab_idx, meas_idx in enumerate(x_stab_meas):
        if stab_idx < hx.shape[0]:
            stab_support = set(np.where(hx[stab_idx])[0])
            overlap = len(stab_support & x_l_support)
            if overlap % 2 == 1:
                frame_meas.append(meas_idx)
    
    return frame_meas


def get_pauli_frame_measurements_for_z_observable(
    code,
    x_stab_meas: List[int],
    z_stab_meas: List[int],
) -> List[int]:
    """
    Determine which projection measurements affect the Z_L observable.
    
    For |0⟩_L, the Z_L eigenvalue depends on Z stabilizer projections
    that overlap with Z_L.
    
    Parameters
    ----------
    code : CSSCode
        The CSS code.
    x_stab_meas : List[int]
        Measurement indices for X stabilizer projections.
    z_stab_meas : List[int]
        Measurement indices for Z stabilizer projections.
        
    Returns
    -------
    List[int]
        Measurement indices that must be XORed into Z_L observable.
    """
    css = code.as_css()
    if css is None:
        return []
    
    lz = np.atleast_2d(css.Lz)
    hz = css.hz
    
    z_l_support = set(np.where(lz[0])[0])
    frame_meas = []
    
    # Check which Z stabilizers have odd overlap with Z_L
    for stab_idx, meas_idx in enumerate(z_stab_meas):
        if stab_idx < hz.shape[0]:
            stab_support = set(np.where(hz[stab_idx])[0])
            overlap = len(stab_support & z_l_support)
            if overlap % 2 == 1:
                frame_meas.append(meas_idx)
    
    return frame_meas


def teleportation_gate_to_injection(gate_name: str) -> InjectionGate:
    """Map teleportation gate name to injection gate."""
    gate_map = {
        "H": InjectionGate.H,
        "HADAMARD": InjectionGate.H,
        "S": InjectionGate.S,
        "S_DAG": InjectionGate.S_DAG,
        "T": InjectionGate.T,
        "T_DAG": InjectionGate.T_DAG,
        "I": InjectionGate.IDENTITY,
        "IDENTITY": InjectionGate.IDENTITY,
    }
    return gate_map.get(gate_name.upper(), InjectionGate.IDENTITY)


