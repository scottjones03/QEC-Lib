"""
Concatenated CSS Code Simulator v10
┌─────────────────────────────────────────────────────────────────────────────┐
│                           CONCATENATED CSS v10                               │
│                                                                              │
│  Combines generalizability (abstract interfaces) with correctness           │
│  (exact Steane implementations matching original)                            │
└─────────────────────────────────────────────────────────────────────────────┘
Combines: 
- Generalizability of v7 (abstract base classes, code-agnostic algorithms)
- Correctness of v9 (exact match to original concatenated_steane.py)

Design principles:
1. CSSCode carries all code-specific circuit details
2. Abstract base classes for gates, EC, and preparation strategies
3. Concrete implementations for Steane that match original exactly
4. General algorithms that work with any CSS code


┌─────────────────────────────────────────────────────────────────────────────┐
│                              DATA LAYER                                      │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────┐     ┌─────────────────────────┐
│        CSSCode          │     │   PropagationTables     │
├─────────────────────────┤     ├─────────────────────────┤
│ Mathematical:            │     │ propagation_X:  List     │
│  - n, k, d              │     │ propagation_Z: List     │
│  - Hz, Hx (stabilizers) │     │ propagation_m: List     │
│  - Lz, Lx (logical ops) │     │ num_ec_0prep: int       │
├─────────────────────────┤     └───────────┬─────────────┘
│ Circuit Specification:  │                 │
│  - h_qubits: [0,1,3]    │                 │
│  - encoding_cnots       │                 │
│  - encoding_cnot_rounds │                 │
│  - verification_qubits  │                 │
│  - idle_schedule        │                 │
└───────────┬─────────────┘                 │
            │                               │
            │ 1.. n                          │ 0.. n (per level)
            ▼                               ▼
┌─────────────────────────────────────────────────────────────┐
│                    ConcatenatedCode                          │
├─────────────────────────────────────────────────────────────┤
│ levels:  List[CSSCode]           # Inner to outer            │
│ propagation_tables: Dict[int, PropagationTables]            │
├─────────────────────────────────────────────────────────────┤
│ + num_levels: int                                           │
│ + total_qubits: int                                         │
│ + qubits_at_level(level) -> int                             │
│ + code_at_level(level) -> CSSCode                           │
│ + get_propagation_tables(level) -> PropagationTables        │
└─────────────────────────────────────────────────────────────┘
"""

import stim
import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Union, Any, Type

from qectostim.noise.models import NoiseModel


# =============================================================================
# Core Data Structures
# =============================================================================

@dataclass
class CSSCode:
    """
    Complete specification of a CSS quantum error correcting code.
    
    Contains both mathematical structure and circuit-level details needed
    for simulation.  The circuit details (h_qubits, encoding_cnots, etc.)
    can be auto-derived for simple cases or explicitly specified for
    exact control.
    """
    # Mathematical structure
    name: str
    n: int  # Number of physical qubits
    k: int  # Number of logical qubits  
    d: int  # Code distance
    Hz: np.ndarray  # Z stabilizer check matrix (detects X errors)
    Hx: np.ndarray  # X stabilizer check matrix (detects Z errors)
    Lz: np.ndarray  # Logical Z operator
    Lx: np.ndarray  # Logical X operator
    
    # Preparation circuit specification
    h_qubits:  List[int] = field(default_factory=list)
    encoding_cnots:  List[Tuple[int, int]] = field(default_factory=list)
    encoding_cnot_rounds: Optional[List[List[Tuple[int, int]]]] = None
    verification_qubits: List[int] = field(default_factory=list)
    
    # Idle qubit schedule for noise modeling:  (round_name, round_idx) -> idle qubits
    idle_schedule:  Optional[Dict[str, List[int]]] = None
    
    def __post_init__(self):
        """Validate and set defaults."""
        assert self.Hz.shape[1] == self.n, f"Hz columns must match n"
        assert self.Hx.shape[1] == self. n, f"Hx columns must match n"
        assert len(self. Lz) == self.n, f"Lz length must match n"
        assert len(self.Lx) == self.n, f"Lx length must match n"
        
        # Auto-derive encoding_cnot_rounds if not provided
        if self.encoding_cnot_rounds is None and self.encoding_cnots:
            self.encoding_cnot_rounds = [[(c, t)] for c, t in self.encoding_cnots]
    
    @property
    def num_x_stabilizers(self) -> int:
        return self.Hx.shape[0]
    
    @property
    def num_z_stabilizers(self) -> int:
        return self.Hz.shape[0]
    
    def get_stabilizer_support(self, stab_type: str, index: int) -> List[int]:
        """Get qubit indices in support of a stabilizer."""
        matrix = self. Hx if stab_type == 'x' else self.Hz
        return [i for i, v in enumerate(matrix[index]) if v == 1]


@dataclass
class PropagationTables:
    """
    Error propagation tables for concatenated code decoding.
    
    These encode how Pauli errors propagate through the state preparation
    circuit and are CRITICAL for correct level-2+ decoding.
    """
    propagation_X: List[List[int]]  # X error propagation per EC round
    propagation_Z: List[List[int]]  # Z error propagation per EC round
    propagation_m: List[int]  # EC rounds affecting verification measurement
    num_ec_0prep: int  # Total EC rounds in preparation circuit


@dataclass
class ConcatenatedCode:
    """Concatenated code with arbitrary levels."""
    levels: List[CSSCode]
    name: Optional[str] = None
    propagation_tables: Dict[int, PropagationTables] = field(default_factory=dict)
    
    def __post_init__(self):
        if self.name is None:
            self.name = "Concat[" + "->".join(c.name for c in self.levels) + "]"
    
    @property
    def num_levels(self) -> int:
        return len(self.levels)
    
    @property
    def total_qubits(self) -> int:
        result = 1
        for code in self.levels:
            result *= code.n
        return result
    
    def qubits_at_level(self, level:  int) -> int:
        """Number of physical qubits in a logical qubit at given level."""
        result = 1
        for i in range(level + 1):
            result *= self.levels[i].n
        return result
    
    def code_at_level(self, level: int) -> CSSCode:
        return self.levels[level]
    
    def get_propagation_tables(self, level: int) -> Optional[PropagationTables]: 
        return self.propagation_tables.get(level)


# =============================================================================
# Result Structures
# ┌─────────────────────────────────────────────────────────────────────────────┐
# │                            RESULT TYPES                                      │
# └─────────────────────────────────────────────────────────────────────────────┘

# ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐
# │   GateResult     │  │   PrepResult     │  │    ECResult      │
# ├──────────────────┤  ├──────────────────┤  ├──────────────────┤
# │ gate_type: str   │  │ level:  int       │  │ level:  int       │
# │ implementation   │  │ detector_0prep   │  │ ec_type: str     │
# │ level: int       │  │ detector_0prep_l2│  │ detector_0prep   │
# │ detectors: List  │  │ detector_X       │  │ detector_0prep_l2│
# │ metadata: Dict   │  │ detector_Z       │  │ detector_X       │
# └──────────────────┘  │ children: List   │  │ detector_Z       │
#                       └──────────────────┘  │ children: List   │
#                                             └──────────────────┘
# =============================================================================

@dataclass
class GateResult:
    """Result of a logical gate application."""
    gate_type: str
    implementation: str
    level: int
    detectors: List = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)


@dataclass
class PrepResult:
    """Result of state preparation - unified structure for all levels."""
    level: int
    detector_0prep: List = field(default_factory=list)
    detector_0prep_l2: Optional[List] = None  # For level-2+
    detector_X:  List = field(default_factory=list)
    detector_Z: List = field(default_factory=list)
    children: List = field(default_factory=list)


@dataclass
class ECResult: 
    """Result of error correction - unified structure for all levels."""
    level: int
    ec_type: str
    detector_0prep: List = field(default_factory=list)
    detector_0prep_l2: Optional[List] = None
    detector_X: List = field(default_factory=list)
    detector_Z: List = field(default_factory=list)
    children: List = field(default_factory=list)


# =============================================================================
# Code Factory Functions
# =============================================================================

# Note: Steane-specific factory functions (create_steane_code, create_steane_propagation_l2,
# create_concatenated_steane) have been moved to concatenated_css_v10_steane.py

def create_shor_code() -> CSSCode:
    """
    Create the [[9,1,3]] Shor code.
    
    The Shor code encodes 1 logical qubit in 9 physical qubits with distance 3.
    It has asymmetric stabilizers: 6 X stabilizers and 2 Z stabilizers.
    
    Encoding circuit:
    1. Start with |ψ⟩|0⟩⁸
    2. Apply H to qubits 0, 3, 6 (creates |+⟩ states)
    3. Apply CNOTs to spread the state:
       - Round 1: CNOT(0,1), CNOT(0,2), CNOT(3,4), CNOT(3,5), CNOT(6,7), CNOT(6,8) [parallel]
       - Round 2: CNOT(0,3), CNOT(0,6) [parallel, could also be sequential]
    """
    Hx = np.array([
        [1, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 1]
    ])
    Hz = np.array([
        [1, 1, 1, 1, 1, 1, 0, 0, 0],
        [0, 0, 0, 1, 1, 1, 1, 1, 1]
    ])
    Lx = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0])
    Lz = np.array([1, 0, 0, 1, 0, 0, 1, 0, 0])
    
    # Encoding CNOT rounds (parallel gates in each round)
    encoding_cnot_rounds = [
        # Round 0: spread within each 3-qubit block (all parallel)
        [(0, 1), (0, 2), (3, 4), (3, 5), (6, 7), (6, 8)],
        # Round 1: connect the blocks
        [(0, 3), (0, 6)]
    ]
    
    # Flatten for encoding_cnots
    encoding_cnots = []
    for round_cnots in encoding_cnot_rounds:
        encoding_cnots.extend(round_cnots)
    
    return CSSCode(
        name="Shor",
        n=9, k=1, d=3,
        Hz=Hz, Hx=Hx,
        Lz=Lz, Lx=Lx,
        h_qubits=[0, 3, 6],
        encoding_cnots=encoding_cnots,
        encoding_cnot_rounds=encoding_cnot_rounds,
        verification_qubits=[0, 3, 6]  # Verify on the control qubits
    )


# Note: create_concatenated_steane has been moved to concatenated_css_v10_steane.py


def create_concatenated_code(codes: List[CSSCode], 
                             propagation_tables: Optional[Dict[int, PropagationTables]] = None) -> ConcatenatedCode:
    """Create a general concatenated code from a list of CSS codes."""
    return ConcatenatedCode(
        levels=codes,
        propagation_tables=propagation_tables or {}
    )


# =============================================================================
# Physical Operations
# ┌─────────────────────────────────────────────────────────────────────────────┐
# │                           OPERATIONS LAYER                                   │
# └─────────────────────────────────────────────────────────────────────────────┘

# ┌─────────────────────────┐         ┌─────────────────────────────────────────┐
# │     PhysicalOps         │         │           TransversalOps                │
# │     (Static Class)      │         ├─────────────────────────────────────────┤
# ├─────────────────────────┤         │ Input:  ConcatenatedCode                 │
# │ + reset(circuit, loc, n)│         ├─────────────────────────────────────────┤
# │ + noisy_reset(...)      │◄────────│ Uses (loc, N_prev, N_now) signature     │
# │ + h(circuit, loc)       │         │ matching original for compatibility     │
# │ + cnot(circuit, c, t)   │         ├─────────────────────────────────────────┤
# │ + noisy_cnot(...)       │         │ + append_h(circuit, loc, N_prev, N_now) │
# │ + swap(circuit, q1, q2) │         │ + append_cnot(...)                      │
# │ + measure(circuit, loc) │         │ + append_noisy_cnot(... , p)             │
# │ + noisy_measure(...)    │         │ + append_swap(...)                      │
# │ + detector(circuit, off)│         │ + append_m(... ) -> List[detectors]      │
# │ + depolarize1(...)      │         │ + append_noisy_m(...) -> List           │
# └─────────────────────────┘         │ + append_noisy_wait(...)                │
#                                     └─────────────────────────────────────────┘
# =============================================================================

class PhysicalOps:
    """Low-level physical qubit operations."""
    
    @staticmethod
    def reset(circuit: stim.Circuit, loc: int, n: int):
        for i in range(n):
            circuit.append("R", loc + i)
    
    @staticmethod
    def noisy_reset(circuit: stim.Circuit, loc: int, n: int, p: float):
        for i in range(n):
            circuit. append("R", loc + i)
        for i in range(n):
            circuit.append("X_ERROR", loc + i, p)
    
    @staticmethod
    def h(circuit: stim.Circuit, loc: int):
        circuit.append("H", loc)
    
    @staticmethod
    def cnot(circuit: stim.Circuit, ctrl: int, targ: int):
        circuit.append("CNOT", [ctrl, targ])
    
    @staticmethod
    def noisy_cnot(circuit: stim.Circuit, ctrl: int, targ: int, p: float):
        circuit.append("CNOT", [ctrl, targ])
        circuit.append("DEPOLARIZE2", [ctrl, targ], p)
    
    @staticmethod
    def swap(circuit: stim.Circuit, q1: int, q2: int):
        circuit.append("SWAP", [q1, q2])
    
    @staticmethod
    def measure(circuit: stim.Circuit, loc: int):
        circuit.append("M", loc)
    
    @staticmethod
    def noisy_measure(circuit: stim.Circuit, loc: int, p: float):
        circuit.append("X_ERROR", loc, p)
        circuit.append("M", loc)
    
    @staticmethod
    def detector(circuit: stim.Circuit, offset: int):
        circuit.append("DETECTOR", stim.target_rec(offset))
    
    @staticmethod
    def depolarize1(circuit: stim.Circuit, loc: int, p: float):
        circuit.append("DEPOLARIZE1", loc, p)


# =============================================================================
# Transversal Operations (General)
# =============================================================================

class TransversalOps:
    """
    Transversal operations matching original function signatures.
    
    The (loc, N_prev, N_now) pattern means:
    - N_prev = 1:  physical level
    - N_prev > 1: operating on encoded blocks of size N_prev
    - N_now = number of qubits/blocks at current level
    """
    
    def __init__(self, concat_code: ConcatenatedCode):
        self.concat_code = concat_code
    
    def block_size(self, level: int) -> int:
        return self.concat_code.qubits_at_level(level)
    
    def append_h(self, circuit: stim. Circuit, loc: int, N_prev: int, N_now: int):
        """Transversal Hadamard."""
        if N_prev == 1:
            for i in range(N_now):
                PhysicalOps.h(circuit, loc + i)
        else:
            for i in range(N_now):
                self.append_h(circuit, (loc + i) * N_prev, 1, N_prev)
    
    def append_cnot(self, circuit: stim.Circuit, loc1: int, loc2: int, 
                    N_prev: int, N_now: int):
        """Transversal CNOT."""
        N = N_prev * N_now
        for i in range(N):
            PhysicalOps.cnot(circuit, loc1 * N_prev + i, loc2 * N_prev + i)
    
    def append_noisy_cnot(self, circuit: stim.Circuit, loc1: int, loc2: int,
                          N_prev: int, N_now: int, p: float):
        """Transversal CNOT with depolarizing noise."""
        N = N_prev * N_now
        for i in range(N):
            PhysicalOps.cnot(circuit, loc1 * N_prev + i, loc2 * N_prev + i)
        for i in range(N):
            circuit.append("DEPOLARIZE2", [loc1 * N_prev + i, loc2 * N_prev + i], p)
    
    def append_swap(self, circuit: stim. Circuit, loc1: int, loc2: int,
                    N_prev: int, N_now: int):
        """Transversal SWAP."""
        for i in range(N_prev * N_now):
            PhysicalOps.swap(circuit, N_prev * loc1 + i, N_prev * loc2 + i)
    
    def append_m(self, circuit: stim. Circuit, loc: int, N_prev: int, N_now:  int,
                 detector_counter: List[int]) -> List: 
        """Transversal measurement with detectors."""
        if N_prev == 1:
            for i in range(N_now):
                PhysicalOps.measure(circuit, loc + i)
            for i in range(N_now):
                PhysicalOps.detector(circuit, i - N_now)
            detector_m = [detector_counter[0], detector_counter[0] + N_now]
            detector_counter[0] += N_now
        else:
            detector_m = [
                self.append_m(circuit, (loc + i) * N_prev, 1, N_prev, detector_counter)
                for i in range(N_now)
            ]
        return detector_m
    
    def append_noisy_m(self, circuit: stim.Circuit, loc: int, N_prev: int,
                       N_now: int, p: float, detector_counter: List[int]) -> List:
        """Transversal measurement with pre-measurement noise."""
        if N_prev == 1:
            for i in range(N_now):
                PhysicalOps.noisy_measure(circuit, loc + i, p)
            for i in range(N_now):
                PhysicalOps.detector(circuit, i - N_now)
            detector_m = [detector_counter[0], detector_counter[0] + N_now]
            detector_counter[0] += N_now
        else:
            detector_m = [
                self.append_noisy_m(circuit, (loc + i) * N_prev, 1, N_prev, p, detector_counter)
                for i in range(N_now)
            ]
        return detector_m
    
    def append_noisy_wait(self, circuit: stim. Circuit, list_loc: List[int],
                          N:  int, p: float, gamma: float, steps: int = 1):
        """Idle noise on qubits."""
        ew = 3/4 * (1 - (1 - 4/3 * gamma) ** steps)
        for loc in list_loc:
            for j in range(N):
                PhysicalOps.depolarize1(circuit, loc + j, ew)


# =============================================================================
# Logical Gate Interface (Abstract)
# ┌─────────────────────────────────────────────────────────────────────────────┐
# │                        LOGICAL GATE HIERARCHY                                │
# └─────────────────────────────────────────────────────────────────────────────┘

#                           ┌─────────────────────┐
#                           │   LogicalGate       │
#                           │     (Abstract)      │
#                           ├─────────────────────┤
#                           │ + gate_name: str    │
#                           │ + implementation    │
#                           │ + block_size(level) │
#                           └──────────┬──────────┘
#                                      │
#            ┌─────────────────────────┼─────────────────────────┐
#            │                         │                         │
#            ▼                         ▼                         ▼
# ┌─────────────────────┐  ┌─────────────────────┐  ┌─────────────────────┐
# │   LogicalHGate      │  │  LogicalCNOTGate    │  │ LogicalMeasurement  │
# │    (Abstract)       │  │    (Abstract)       │  │    (Abstract)       │
# ├─────────────────────┤  ├─────────────────────┤  ├─────────────────────┤
# │ + apply(circuit,    │  │ + apply(circuit,    │  │ + apply(circuit,    │
# │   loc, level,       │  │   loc_ctrl, loc_targ│  │   loc, level,       │
# │   detector_counter) │  │   level, det_ctr)   │  │   det_ctr, basis)   │
# │   -> GateResult     │  │   -> GateResult     │  │   -> GateResult     │
# └──────────┬──────────┘  └──────────┬──────────┘  └──────────┬──────────┘
#            │                        │                        │
#            ▼                        ▼                        ▼
# ┌─────────────────────┐  ┌─────────────────────┐  ┌─────────────────────┐
# │ TransversalHGate    │  │ TransversalCNOTGate │  │TransversalMeasure-  │
# │                     │  │                     │  │ment                 │
# ├─────────────────────┤  ├─────────────────────┤  ├─────────────────────┤
# │ Uses TransversalOps │  │ Uses TransversalOps │  │ Uses TransversalOps │
# │ to apply H to all   │  │ to apply CNOT       │  │ Hierarchical        │
# │ physical qubits     │  │ block-wise          │  │ detector structure  │
# └─────────────────────┘  └─────────────────────┘  └─────────────────────┘


#                     ┌─────────────────────────────────┐
#                     │    LogicalGateDispatcher        │
#                     ├─────────────────────────────────┤
#                     │ Input: ConcatenatedCode,        │
#                     │        TransversalOps           │
#                     ├─────────────────────────────────┤
#                     │ _h:  LogicalHGate                │
#                     │ _cnot: LogicalCNOTGate          │
#                     │ _measure: LogicalMeasurement    │
#                     ├─────────────────────────────────┤
#                     │ + h(circuit, loc, level, ctr)   │
#                     │ + cnot(circuit, c, t, lvl, ctr) │
#                     │ + measure(circuit, loc, ...)    │
#                     │ + set_h_gate(gate)              │
#                     │ + set_cnot_gate(gate)           │
#                     └─────────────────────────────────┘
# =============================================================================

class LogicalGate(ABC):
    """Abstract base class for logical gate implementations."""
    
    def __init__(self, concat_code: ConcatenatedCode):
        self.concat_code = concat_code
    
    @property
    @abstractmethod
    def gate_name(self) -> str:
        pass
    
    @property
    @abstractmethod
    def implementation_name(self) -> str:
        pass
    
    def block_size(self, level: int) -> int:
        return self.concat_code.qubits_at_level(level)


class LogicalHGate(LogicalGate):
    """Abstract base for logical Hadamard."""
    
    @property
    def gate_name(self) -> str:
        return "H"
    
    @abstractmethod
    def apply(self, circuit: stim. Circuit, loc: int, level: int,
              detector_counter: List[int]) -> GateResult:
        pass


class LogicalCNOTGate(LogicalGate):
    """Abstract base for logical CNOT."""
    
    @property
    def gate_name(self) -> str:
        return "CNOT"
    
    @abstractmethod
    def apply(self, circuit: stim. Circuit, loc_ctrl: int, loc_targ:  int,
              level: int, detector_counter: List[int]) -> GateResult:
        pass


class LogicalMeasurement(LogicalGate):
    """Abstract base for logical measurement."""
    
    @property
    def gate_name(self) -> str:
        return "MEASURE"
    
    @abstractmethod
    def apply(self, circuit: stim.Circuit, loc: int, level: int,
              detector_counter: List[int], basis: str = 'z') -> GateResult:
        pass


# =============================================================================
# Transversal Gate Implementations
# =============================================================================

class TransversalHGate(LogicalHGate):
    """Transversal Hadamard - applies H to every physical qubit."""
    
    def __init__(self, concat_code: ConcatenatedCode, ops: TransversalOps):
        super().__init__(concat_code)
        self.ops = ops
    
    @property
    def implementation_name(self) -> str:
        return "transversal"
    
    def apply(self, circuit, loc, level, detector_counter):
        N_prev = self.block_size(level - 1) if level > 0 else 1
        N_now = self.concat_code.code_at_level(level).n
        self.ops.append_h(circuit, loc, N_prev, N_now)
        return GateResult(self.gate_name, self.implementation_name, level)


class TransversalCNOTGate(LogicalCNOTGate):
    """Transversal CNOT - applies CNOT between corresponding qubits."""
    
    def __init__(self, concat_code: ConcatenatedCode, ops: TransversalOps):
        super().__init__(concat_code)
        self.ops = ops
    
    @property
    def implementation_name(self) -> str:
        return "transversal"
    
    def apply(self, circuit, loc_ctrl, loc_targ, level, detector_counter):
        N_prev = self.block_size(level - 1) if level > 0 else 1
        N_now = self.concat_code.code_at_level(level).n
        self.ops.append_cnot(circuit, loc_ctrl, loc_targ, N_prev, N_now)
        return GateResult(self.gate_name, self.implementation_name, level)


class TransversalMeasurement(LogicalMeasurement):
    """Transversal measurement - measures all physical qubits."""
    
    def __init__(self, concat_code: ConcatenatedCode, ops: TransversalOps):
        super().__init__(concat_code)
        self.ops = ops
    
    @property
    def implementation_name(self) -> str:
        return "transversal"
    
    def apply(self, circuit, loc, level, detector_counter, basis='z'):
        N_prev = self.block_size(level - 1) if level > 0 else 1
        N_now = self.concat_code.code_at_level(level).n
        
        if basis == 'x':
            self.ops.append_h(circuit, loc, N_prev, N_now)
        
        detectors = self.ops.append_m(circuit, loc, N_prev, N_now, detector_counter)
        
        result = GateResult(self.gate_name, self.implementation_name, level)
        result.detectors = detectors
        return result


# =============================================================================
# Gate Dispatcher
# =============================================================================

class LogicalGateDispatcher:
    """
    Dispatcher for logical gate operations.
    
    Provides unified interface abstracting gate implementations.
    """
    
    def __init__(self, concat_code: ConcatenatedCode, ops: TransversalOps):
        self.concat_code = concat_code
        self.ops = ops
        
        # Default to transversal implementations
        self._h = TransversalHGate(concat_code, ops)
        self._cnot = TransversalCNOTGate(concat_code, ops)
        self._measure = TransversalMeasurement(concat_code, ops)
    
    def set_h_gate(self, gate: LogicalHGate):
        self._h = gate
    
    def set_cnot_gate(self, gate: LogicalCNOTGate):
        self._cnot = gate
    
    def set_measurement(self, gate: LogicalMeasurement):
        self._measure = gate
    
    def h(self, circuit: stim.Circuit, loc: int, level: int,
          detector_counter: List[int]) -> GateResult:
        return self._h.apply(circuit, loc, level, detector_counter)
    
    def cnot(self, circuit: stim.Circuit, loc_ctrl: int, loc_targ: int,
             level: int, detector_counter: List[int]) -> GateResult:
        return self._cnot.apply(circuit, loc_ctrl, loc_targ, level, detector_counter)
    
    def measure(self, circuit: stim.Circuit, loc: int, level: int,
                detector_counter: List[int], basis: str = 'z') -> GateResult:
        return self._measure.apply(circuit, loc, level, detector_counter, basis)


# =============================================================================
# Preparation Strategy (Abstract)
# ┌─────────────────────────────────────────────────────────────────────────────┐
# │                       PREPARATION STRATEGIES                                 │
# └─────────────────────────────────────────────────────────────────────────────┘

#                     ┌─────────────────────────────────┐
#                     │    PreparationStrategy          │
#                     │         (Abstract)              │
#                     ├─────────────────────────────────┤
#                     │ Input: ConcatenatedCode,        │
#                     │        TransversalOps           │
#                     ├─────────────────────────────────┤
#                     │ + set_ec_gadget(ECGadget)       │
#                     │ + strategy_name: str            │
#                     ├─────────────────────────────────┤
#                     │ + append_0prep(circuit, loc1,   │
#                     │     N_prev, N_now)              │
#                     │                                 │
#                     │ + append_noisy_0prep(circuit,   │
#                     │     loc1, loc2, N_prev, N_now,  │
#                     │     p, detector_counter)        │
#                     │   -> List | Tuple               │
#                     └────────────────┬────────────────┘
#                                      │
#                     ┌────────────────┴────────────────┐
#                     │                                 │
#                     ▼                                 ▼
#      ┌──────────────────────────┐      ┌──────────────────────────┐
#      │ SteanePreparationStrategy│      │GenericPreparationStrategy│
#      ├──────────────────────────┤      ├──────────────────────────┤
#      │ Matches original EXACTLY │      │ Uses CSSCode spec to     │
#      │                          │      │ build preparation        │
#      │ _noisy_0prep_l1():       │      │                          │
#      │  - 3 H gates             │      │ Less optimized but       │
#      │  - 8 encoding CNOTs      │      │ works for any CSS code   │
#      │  - 3 verification CNOTs  │      │                          │
#      │  - Idle noise            │      │ No EC interleaving       │
#      │  - 1 measurement         │      │ (simpler structure)      │
#      │                          │      │                          │
#      │ _noisy_0prep_l2():       │      └──────────────────────────┘
#      │  - Recursive inner prep  │
#      │  - 45 EC rounds          │
#      │  - Exact interleaving    │
#      │  - Returns 4-tuple       │
#      └──────────────────────────┘

# Return Value Structure: 
# ┌─────────────────────────────────────────────────────────────────┐
# │ N_prev = 1 (Level-1):                                           │
# │   Returns:  detector_0prep (List)                                │
# │                                                                 │
# │ N_prev > 1, N_now = N_steane (Level-2):                         │
# │   Returns: (detector_0prep, detector_0prep_l2,                  │
# │             detector_X, detector_Z)                             │
# └─────────────────────────────────────────────────────────────────┘
# =============================================================================

class PreparationStrategy(ABC):
    """
    Abstract base class for state preparation strategies.
    
    Different codes may require different preparation circuits with
    different EC interleaving patterns.
    """
    
    def __init__(self, concat_code: ConcatenatedCode, ops: TransversalOps):
        self.concat_code = concat_code
        self.ops = ops
        self._ec_gadget = None
    
    def set_ec_gadget(self, ec_gadget: 'ECGadget'):
        self._ec_gadget = ec_gadget
    
    @property
    def ec(self) -> 'ECGadget':
        if self._ec_gadget is None:
            raise RuntimeError("EC gadget not set")
        return self._ec_gadget
    
    @property
    @abstractmethod
    def strategy_name(self) -> str:
        pass
    
    @property
    def uses_prep_ec_at_l2(self) -> bool:
        """
        Whether this preparation strategy uses EC rounds during L2 preparation.
        
        This affects the detector structure returned:
        - True: detector_X/detector_Z contain prep EC entries (old-style)
        - False: detector_X/detector_Z are empty (corrected prep style)
        
        Subclasses should override if they use prep EC at L2.
        Default is False (corrected prep style with no EC during prep).
        """
        return False
    
    @abstractmethod
    def append_0prep(self, circuit: stim.Circuit, loc1: int, 
                     N_prev: int, N_now: int):
        """Noiseless |0⟩_L preparation."""
        pass
    
    @abstractmethod
    def append_noisy_0prep(self, circuit: stim.Circuit, loc1: int, loc2: int,
                           N_prev: int, N_now: int, p: float,
                           detector_counter: List[int]) -> Union[List, Tuple]:
        """
        Noisy |0⟩_L preparation with verification.
        
        Returns:
        - N_prev=1: detector_0prep (list)
        - N_prev>1: (detector_0prep, detector_0prep_l2, detector_X, detector_Z)
        """
        pass


# Note: SteanePreparationStrategy has been moved to concatenated_css_v10_steane.py

class GenericPreparationStrategy(PreparationStrategy):
    """
    Generic preparation strategy that works for any CSS code.
    
    Uses the CSSCode specification to build preparation circuits that are
    as close as possible to optimal code-specific implementations.
    
    Key features:
    - Uses code's h_qubits, encoding_cnots, encoding_cnot_rounds
    - Applies EC interleaving based on CNOT rounds (if specified)
    - Handles idle noise on non-active qubits
    - Supports both level-1 and level-2 preparation
    - Returns correct structure for decoder compatibility
    """
    
    @property
    def strategy_name(self) -> str:
        return "generic"
    
    def append_0prep(self, circuit: stim.Circuit, loc1: int,
                     N_prev: int, N_now: int):
        """
        Noiseless |0⟩_L preparation using code specification.
        
        Recursively prepares inner blocks, then applies encoding circuit.
        """
        code = self.concat_code. code_at_level(0)
        
        # Base case: physical qubits
        if N_prev == 1:
            PhysicalOps.reset(circuit, loc1, N_now)
        else:
            # Recursive case: prepare each inner block
            for i in range(N_now):
                self.append_0prep(circuit, (loc1 + i) * N_prev, 1, N_prev)
        
        # Apply encoding circuit if at code block size
        if N_now == code.n: 
            # H gates on specified qubits
            for q in code.h_qubits:
                self.ops.append_h(circuit, (loc1 + q) * N_prev, 1, N_prev)
            
            # CNOTs in specified order
            for ctrl, targ in code.encoding_cnots:
                self.ops.append_cnot(circuit, (loc1 + ctrl) * N_prev,
                                     (loc1 + targ) * N_prev, 1, N_prev)
    
    def append_noisy_0prep(self, circuit: stim.Circuit, loc1: int, loc2: int,
                           N_prev: int, N_now: int, p:  float,
                           detector_counter: List[int]) -> Union[List, Tuple]:
        """
        Noisy |0⟩_L preparation with verification.
        
        Dispatches to level-specific implementations based on N_prev.
        Returns different structures for L1 vs L2 to match decoder expectations.
        """
        code = self.concat_code.code_at_level(0)
        gamma = self._compute_gamma(p)
        
        # Prepare inner blocks recursively
        if N_prev == 1:
            PhysicalOps.noisy_reset(circuit, loc1, N_now, p)
            PhysicalOps.noisy_reset(circuit, loc2, N_now, p)
            detector_0prep = []
        else:
            detector_0prep = []
            # Prepare data blocks
            for i in range(N_now):
                result = self.append_noisy_0prep(
                    circuit, (loc1 + i) * N_prev, (loc1 + N_now + i) * N_prev,
                    1, N_prev, p, detector_counter
                )
                detector_0prep.append(result)
            
            # Prepare verification/ancilla block(s)
            if N_now == code.n:
                # Single verification block for code-sized preparation
                result = self.append_noisy_0prep(
                    circuit, loc2 * N_prev, (loc2 + N_now) * N_prev,
                    1, N_prev, p, detector_counter
                )
                detector_0prep.append(result)
            else:
                # Multiple ancilla blocks for non-code-sized
                for i in range(N_now):
                    result = self.append_noisy_0prep(
                        circuit, (loc2 + i) * N_prev, (loc2 + N_now + i) * N_prev,
                        1, N_prev, p, detector_counter
                    )
                    detector_0prep.append(result)
        
        # Apply encoding based on level
        if N_now == code. n:
            if N_prev != 1:
                return self._noisy_0prep_l2(
                    circuit, loc1, loc2, N_prev, N_now, p, gamma,
                    detector_counter, detector_0prep, code
                )
            else:
                return self._noisy_0prep_l1(
                    circuit, loc1, loc2, N_prev, N_now, p, gamma,
                    detector_counter, detector_0prep, code
                )
        
        return detector_0prep
    
    def _compute_gamma(self, p: float) -> float:
        """Compute idle error rate from physical error rate."""
        # Default:  error model 'a' from original
        return p / 10
    
    def _get_idle_qubits(self, code: CSSCode, active_qubits: List[int], 
                         round_name: str = None) -> List[int]:
        """
        Determine which qubits are idle given active qubits. 
        
        Uses code's idle_schedule if available, otherwise computes from active set.
        """
        all_qubits = set(range(code.n))
        active_set = set(active_qubits)
        idle = list(all_qubits - active_set)
        
        # Override with code-specific schedule if available
        if code.idle_schedule and round_name and round_name in code.idle_schedule:
            idle = code. idle_schedule[round_name]
        
        return sorted(idle)
    
    def _get_cnot_rounds(self, code: CSSCode) -> List[List[Tuple[int, int]]]: 
        """
        Get CNOT rounds from code specification.
        
        Uses encoding_cnot_rounds if available, otherwise groups by parallelizability.
        """
        if code.encoding_cnot_rounds:
            return code.encoding_cnot_rounds
        
        # Auto-group CNOTs that can be parallelized
        # CNOTs are parallel if they don't share any qubits
        if not code.encoding_cnots:
            return []
        
        rounds = []
        remaining = list(code.encoding_cnots)
        
        while remaining: 
            current_round = []
            used_qubits = set()
            still_remaining = []
            
            for ctrl, targ in remaining:
                if ctrl not in used_qubits and targ not in used_qubits:
                    current_round.append((ctrl, targ))
                    used_qubits.add(ctrl)
                    used_qubits.add(targ)
                else:
                    still_remaining. append((ctrl, targ))
            
            if current_round: 
                rounds.append(current_round)
            remaining = still_remaining
        
        return rounds
    
    def _get_verification_schedule(self, code: CSSCode) -> List[List[int]]:
        """
        Get verification CNOT schedule. 
        
        Groups verification qubits for sequential verification CNOTs.
        For fault-tolerance, typically one verification qubit per round.
        
        If verification_qubits is not specified in the code, derives them
        from the code structure (uses qubits in the logical X support).
        """
        verif_qubits = code.verification_qubits
        
        # Auto-derive verification qubits if not specified
        if not verif_qubits:
            # Use qubits in logical X operator support
            # These are the qubits that determine the logical state in Z-basis
            verif_qubits = [i for i, v in enumerate(code.Lx) if v == 1]
            
            # If Lx is weight-n (all qubits), fall back to h_qubits
            if len(verif_qubits) == code.n and code.h_qubits:
                verif_qubits = code.h_qubits
            
            # Last resort: use first ceil(n/2) qubits
            if not verif_qubits:
                verif_qubits = list(range((code.n + 1) // 2))
        
        # Default: one verification qubit per round (most fault-tolerant)
        return [[vq] for vq in verif_qubits]
    
    def _noisy_0prep_l1(self, circuit: stim.Circuit, loc1: int, loc2: int,
                        N_prev: int, N_now: int, p: float, gamma: float,
                        detector_counter: List[int], detector_0prep: List,
                        code: CSSCode) -> List:
        """
        Level-1 noisy preparation (N_prev=1).
        
        Structure: 
        1. H gates on designated qubits
        2. Encoding CNOTs (by rounds with idle noise)
        3. Verification CNOTs (with idle noise)
        4. Verification measurement
        """
        n = code.n
        
        # Phase 1: H gates
        for q in code.h_qubits:
            self.ops. append_h(circuit, (loc1 + q) * N_prev, 1, N_prev)
        
        # Phase 2: Encoding CNOTs by rounds
        cnot_rounds = self._get_cnot_rounds(code)
        
        for round_idx, round_cnots in enumerate(cnot_rounds):
            # Determine active qubits in this round
            active_qubits = []
            for ctrl, targ in round_cnots:
                active_qubits.extend([ctrl, targ])
            
            # Apply CNOTs
            for ctrl, targ in round_cnots:
                self. ops.append_noisy_cnot(
                    circuit, (loc1 + ctrl) * N_prev, (loc1 + targ) * N_prev,
                    1, N_prev, p
                )
            
            # Apply idle noise to non-active qubits
            idle_qubits = self._get_idle_qubits(code, active_qubits, f'cnot_round_{round_idx}')
            if idle_qubits:
                self.ops.append_noisy_wait(
                    circuit, [(loc1 + q) * N_prev for q in idle_qubits],
                    N_prev, p, gamma, steps=1
                )
        
        # Phase 3: Verification CNOTs
        verification_schedule = self._get_verification_schedule(code)
        
        for verif_idx, verif_qubits in enumerate(verification_schedule):
            # Apply verification CNOTs
            for vq in verif_qubits:
                self.ops.append_noisy_cnot(
                    circuit, (loc1 + vq) * N_prev, loc2 * N_prev,
                    1, N_prev, p
                )
            
            # Idle noise on non-verification qubits
            idle_qubits = self._get_idle_qubits(code, verif_qubits, f'verif_cnot_{verif_idx}')
            if idle_qubits: 
                self.ops.append_noisy_wait(
                    circuit, [(loc1 + q) * N_prev for q in idle_qubits],
                    N_prev, p, gamma, steps=1
                )
        
        # Phase 4: Verification measurement
        detector_0prep.append(
            self.ops.append_noisy_m(circuit, loc2 * N_prev, 1, N_prev, p, detector_counter)
        )
        
        # Idle noise during measurement
        # Get all verification qubits from schedule
        all_verif_qubits = []
        for vq_list in verification_schedule:
            all_verif_qubits.extend(vq_list)
        
        if all_verif_qubits:
            non_measured = [q for q in range(n) if q not in all_verif_qubits[-1:]]
            if non_measured:
                self.ops.append_noisy_wait(
                    circuit, [(loc1 + q) * N_prev for q in non_measured],
                    N_prev, p, gamma, steps=1
                )
        
        return detector_0prep
    
    def _noisy_0prep_l2(self, circuit: stim.Circuit, loc1: int, loc2: int,
                        N_prev: int, N_now: int, p: float, gamma: float,
                        detector_counter: List[int], detector_0prep: List,
                        code: CSSCode) -> Tuple:
        """
        Level-2 noisy preparation (N_prev > 1, N_now = code.n).
        
        CORRECTED STRUCTURE (no EC during prep to avoid noise amplification):
        1. H gates on designated qubits
        2. Encoding CNOTs (by rounds with idle noise only)
        3. Verification CNOTs (with idle noise only)
        4. Verification measurement
        5. Decorrelation CNOTs (undo entanglement with measured ancilla)
        
        Returns 4-tuple: (detector_0prep, detector_0prep_l2, [], [])
        Note: detector_X and detector_Z are empty since no EC during prep.
        """
        n = code.n
        
        # NO EC during L2 prep - EC amplifies noise instead of helping
        # We only do: H gates -> encoding CNOTs -> verification CNOTs -> verification measurement -> decorrelation CNOTs
        
        # Phase 1: H gates on designated qubits
        for q in code.h_qubits:
            self.ops.append_h(circuit, (loc1 + q) * N_prev, 1, N_prev)
        
        # Phase 2: Encoding CNOTs by rounds (NO EC)
        cnot_rounds = self._get_cnot_rounds(code)
        
        for round_idx, round_cnots in enumerate(cnot_rounds):
            # Determine active qubits for idle noise
            active_qubits = []
            for ctrl, targ in round_cnots:
                active_qubits.extend([ctrl, targ])
            
            # Apply CNOTs
            for ctrl, targ in round_cnots:
                self.ops.append_noisy_cnot(
                    circuit, (loc1 + ctrl) * N_prev, (loc1 + targ) * N_prev,
                    1, N_prev, p
                )
            
            # Idle noise on non-active qubits
            idle_qubits = self._get_idle_qubits(code, active_qubits, f'cnot_round_{round_idx}')
            if idle_qubits:
                self.ops.append_noisy_wait(
                    circuit, [(loc1 + q) * N_prev for q in idle_qubits],
                    N_prev, p, gamma, steps=1
                )
        
        # Phase 3: Verification CNOTs (NO EC)
        verification_schedule = self._get_verification_schedule(code)
        
        for verif_idx, verif_qubits in enumerate(verification_schedule):
            # Apply verification CNOTs from data to verification ancilla
            for vq in verif_qubits:
                self.ops.append_noisy_cnot(
                    circuit, (loc1 + vq) * N_prev, loc2 * N_prev,
                    1, N_prev, p
                )
            
            # Idle noise on non-active qubits
            idle_qubits = self._get_idle_qubits(code, verif_qubits, f'verif_cnot_{verif_idx}')
            if idle_qubits:
                self.ops.append_noisy_wait(
                    circuit, [(loc1 + q) * N_prev for q in idle_qubits],
                    N_prev, p, gamma, steps=1
                )
        
        # Phase 4: Verification measurement
        detector_0prep_l2 = self.ops.append_noisy_m(
            circuit, loc2 * N_prev, 1, N_prev, p, detector_counter
        )
        
        # Phase 5: Decorrelation CNOTs
        # After measuring verification ancilla, apply CNOTs to undo entanglement with measured ancilla
        # Use the same verification qubits derived from the schedule
        all_verif_qubits = []
        for verif_qubits in verification_schedule:
            all_verif_qubits.extend(verif_qubits)
        
        for vq in all_verif_qubits:
            self.ops.append_cnot(circuit, loc2 * N_prev, (loc1 + vq) * N_prev, 1, N_prev)
        
        # Return empty detector_X and detector_Z since no EC during prep
        return detector_0prep, detector_0prep_l2, [], []
    
    def _get_initial_ec_qubits(self, code: CSSCode) -> List[int]:
        """
        Determine which qubits should have EC after H gates.
        
        Strategy: EC on qubits that are active early in the circuit
        to catch errors before they propagate. This includes:
        - Qubits with H gates applied
        - Control qubits of the first CNOT round  
        - Target qubits of the first CNOT round
        
        For small codes (n <= 4), include all qubits.
        For larger codes, exclude the last qubit if it's not used until later
        (allows encoding to proceed without all qubits being initialized).
        """
        initial_qubits = set(code.h_qubits)
        
        # Add all qubits involved in first CNOT round
        cnot_rounds = self._get_cnot_rounds(code)
        if cnot_rounds:
            for ctrl, targ in cnot_rounds[0]:
                initial_qubits.add(ctrl)
                initial_qubits.add(targ)
        
        # For small codes, use all qubits
        if code.n <= 4:
            return list(range(code.n))
        
        # For larger codes, sort and filter
        result = sorted(initial_qubits)
        
        # If we have very few initial qubits, expand to include more
        # At minimum, include all h_qubits and their partners
        min_qubits = max(len(code.h_qubits), code.n // 2)
        if len(result) < min_qubits:
            result = list(range(min_qubits))
        
        return result
    
    def compute_propagation_tables(self, code: CSSCode) -> PropagationTables:
        """
        Compute propagation tables for this preparation strategy.
        
        This tracks how errors propagate through the encoding circuit:
        - X errors propagate forward through CNOT controls to targets
        - Z errors propagate backward through CNOT targets to controls
        
        For each EC measurement location in the 0-prep circuit, we track
        which data qubits are affected by X and Z errors at that location.
        
        Returns tables describing how errors propagate through the circuit.
        """
        n = code.n
        cnot_rounds = self._get_cnot_rounds(code)
        verification_schedule = self._get_verification_schedule(code)
        
        # Build propagation tables by simulating error flow
        propagation_X = []
        propagation_Z = []
        
        # Track which qubits each qubit's errors affect AFTER all subsequent CNOTs
        # We build this incrementally as we process CNOTs
        
        # For X propagation: X on qubit q before CNOT(c,t) -> X on q (and also t if q=c)
        # For Z propagation: Z on qubit q before CNOT(c,t) -> Z on q (and also c if q=t)
        
        # Process the circuit in order to build propagation
        # Initial EC: on h_qubits and first-round targets
        initial_ec_qubits = self._get_initial_ec_qubits(code)
        
        # Build future CNOT list for propagation calculation
        all_cnots = []
        for round_cnots in cnot_rounds:
            all_cnots.extend(round_cnots)
        
        def get_x_propagation(qubit: int, remaining_cnots: List[Tuple[int, int]]) -> List[int]:
            """Calculate which qubits an X error on 'qubit' propagates to."""
            affected = {qubit}
            for ctrl, targ in remaining_cnots:
                if ctrl in affected:
                    affected.add(targ)
            return sorted(affected - {qubit})  # Exclude self
        
        def get_z_propagation(qubit: int, remaining_cnots: List[Tuple[int, int]]) -> List[int]:
            """Calculate which qubits a Z error on 'qubit' propagates to."""
            affected = {qubit}
            for ctrl, targ in remaining_cnots:
                if targ in affected:
                    affected.add(ctrl)
            return sorted(affected - {qubit})  # Exclude self
        
        # EC rounds counter
        ec_round = 0
        cnots_processed = 0
        
        # Initial EC rounds (after H gates, before first CNOT round)
        for q in initial_ec_qubits:
            remaining = all_cnots[cnots_processed:]
            propagation_X.append(get_x_propagation(q, remaining))
            propagation_Z.append(get_z_propagation(q, remaining))
            ec_round += 1
        
        # EC after each CNOT round
        for round_idx, round_cnots in enumerate(cnot_rounds):
            cnots_processed += len(round_cnots)
            remaining = all_cnots[cnots_processed:]
            
            # EC on all n data qubits after this round
            for q in range(n):
                propagation_X.append(get_x_propagation(q, remaining))
                propagation_Z.append(get_z_propagation(q, remaining))
                ec_round += 1
        
        # EC on verification ancilla (ancilla doesn't propagate to data)
        propagation_X.append([])
        propagation_Z.append([])
        ec_round += 1
        
        # Verification rounds (after all encoding CNOTs)
        for verif_idx, verif_qubits in enumerate(verification_schedule):
            # After verification CNOT, EC on data qubits
            # Errors at this point don't propagate further (encoding done)
            for q in range(n):
                propagation_X.append([])
                propagation_Z.append([])
                ec_round += 1
            
            # EC on verification ancilla
            propagation_X.append([])
            propagation_Z.append([])
            ec_round += 1
        
        num_ec_0prep = ec_round
        
        # Propagation to measurement: which EC rounds affect the verification measurement
        # An EC round affects measurement if X errors propagate to verification qubits
        propagation_m = []
        for ec_idx in range(num_ec_0prep):
            if ec_idx < len(propagation_X):
                # Check if any verification qubit is in the X propagation
                for vq in code.verification_qubits:
                    if vq in propagation_X[ec_idx]:
                        propagation_m.append(ec_idx)
                        break
        
        propagation_m = sorted(set(propagation_m))
        
        return PropagationTables(
            propagation_X=propagation_X,
            propagation_Z=propagation_Z,
            propagation_m=propagation_m,
            num_ec_0prep=num_ec_0prep
        )

# =============================================================================
# EC Gadget (Abstract and Implementations)
# ┌─────────────────────────────────────────────────────────────────────────────┐
# │                          EC GADGET HIERARCHY                                 │
# └─────────────────────────────────────────────────────────────────────────────┘

#                     ┌─────────────────────────────────┐
#                     │         ECGadget                │
#                     │         (Abstract)              │
#                     ├─────────────────────────────────┤
#                     │ Input: ConcatenatedCode,        │
#                     │        TransversalOps           │
#                     ├─────────────────────────────────┤
#                     │ + set_prep(PreparationStrategy) │
#                     │ + ec_type: str                  │
#                     ├─────────────────────────────────┤
#                     │ + append_noisy_ec(circuit,      │
#                     │     loc1, loc2, loc3, loc4,     │
#                     │     N_prev, N_now, p, det_ctr)  │
#                     │   -> Tuple                      │
#                     └────────────────┬────────────────┘
#                                      │
#                     ┌────────────────┴────────────────┐
#                     │                                 │
#                     ▼                                 ▼
#      ┌──────────────────────────┐      ┌──────────────────────────┐
#      │     SteaneECGadget       │      │     KnillECGadget        │
#      ├──────────────────────────┤      ├──────────────────────────┤
#      │ ec_type = "steane"       │      │ ec_type = "knill"        │
#      │                          │      │                          │
#      │ Structure:               │      │ Structure:                │
#      │ 1. Prepare 2 ancillas    │      │ 1. Prepare Bell pair     │
#      │ 2. H on ancilla 1        │      │ 2. Bell measurement      │
#      │ 3.  CNOT ancillas         │      │ 3. Teleport state        │
#      │ 4. Recursive EC (L2)     │      │                          │
#      │ 5.  CNOT data->ancilla    │      │ Uses teleportation       │
#      │ 6. H on data             │      │ for error correction     │
#      │ 7. Measure syndromes     │      │                          │
#      │ 8.  SWAP corrected state  │      │                          │
#      └──────────────────────────┘      └──────────────────────────┘

# Return Value Structure:
# ┌─────────────────────────────────────────────────────────────────┐
# │ N_prev = 1:                                                     │
# │   Returns: (detector_0prep, detector_Z, detector_X)             │
# │                                                                 │
# │ N_prev > 1:                                                     │
# │   Returns: (detector_0prep, detector_0prep_l2,                  │
# │             detector_Z, detector_X)                             │
# └─────────────────────────────────────────────────────────────────┘

# Circular Dependency Resolution:
# ┌─────────────────────────────────────────────────────────────────┐
# │                                                                 │
# │  PreparationStrategy ◄─────────────► ECGadget                   │
# │         │                                 │                     │
# │         │ set_ec_gadget()                 │ set_prep()          │
# │         └─────────────────────────────────┘                     │
# │                                                                 │
# │  Both need each other because:                                  │
# │  - Preparation uses EC for level-2                              │
# │  - EC uses Preparation for ancilla states                       │
# │                                                                 │
# └─────────────────────────────────────────────────────────────────┘
# =============================================================================

class ECGadget(ABC):
    """Abstract base class for error correction gadgets."""
    
    def __init__(self, concat_code: ConcatenatedCode, ops: TransversalOps):
        self.concat_code = concat_code
        self.ops = ops
        self._prep = None
    
    def set_prep(self, prep: PreparationStrategy):
        self._prep = prep
    
    @property
    def prep(self) -> PreparationStrategy:
        if self._prep is None:
            raise RuntimeError("Preparation strategy not set")
        return self._prep
    
    @property
    @abstractmethod
    def ec_type(self) -> str:
        pass
    
    @abstractmethod
    def append_noisy_ec(self, circuit:  stim.Circuit, loc1: int, loc2: int,
                        loc3: int, loc4: int, N_prev: int, N_now: int,
                        p: float, detector_counter:  List[int]) -> Tuple:
        """
        Apply EC gadget. 
        
        Returns:
        - N_prev=1: (detector_0prep, detector_Z, detector_X)
        - N_prev>1: (detector_0prep, detector_0prep_l2, detector_Z, detector_X)
        """
        pass


# Note: SteaneECGadget has been moved to concatenated_css_v10_steane.py


class KnillECGadget(ECGadget):
    """Knill-style teleportation EC."""
    
    @property
    def ec_type(self) -> str:
        return "knill"
    
    def append_noisy_ec(self, circuit: stim.Circuit, loc1: int, loc2: int,
                        loc3: int, loc4: int, N_prev: int, N_now: int,
                        p: float, detector_counter: List[int]) -> Tuple:
        """Knill EC using teleportation."""
        # Similar structure to Steane but uses teleportation
        detector_0prep = []
        detector_0prep_l2 = []
        detector_Z = []
        detector_X = []
        
        if N_now == 1:
            return None
        
        n_now = N_now
        
        # Prepare Bell pair
        if N_prev == 1:
            result1 = self.prep.append_noisy_0prep(circuit, loc2, loc4, N_prev, N_now, p, detector_counter)
            result2 = self.prep.append_noisy_0prep(circuit, loc3, loc4, N_prev, N_now, p, detector_counter)
            detector_0prep.extend(result1)
            detector_0prep.extend(result2)
        else:
            result1 = self.prep. append_noisy_0prep(circuit, loc2, loc4, N_prev, N_now, p, detector_counter)
            detector_0prep.extend(result1[0])
            detector_0prep_l2.append(result1[1])
            detector_X.extend(result1[2])
            detector_Z. extend(result1[3])
            
            result2 = self.prep.append_noisy_0prep(circuit, loc3, loc4, N_prev, N_now, p, detector_counter)
            detector_0prep. extend(result2[0])
            detector_0prep_l2.append(result2[1])
            detector_X.extend(result2[2])
            detector_Z.extend(result2[3])
        
        # Create Bell pair
        self.ops.append_h(circuit, loc2, N_prev, n_now)
        self.ops.append_noisy_cnot(circuit, loc2, loc3, N_prev, n_now, p)
        
        # Bell measurement
        self.ops.append_noisy_cnot(circuit, loc1, loc2, N_prev, n_now, p)
        self.ops.append_h(circuit, loc1, N_prev, n_now)
        
        # Measure - loc1 gives Z syndrome, loc2 gives X syndrome
        # (matching SteaneECGadget convention)
        detector_Z.append(self.ops.append_noisy_m(circuit, loc1, N_prev, n_now, p, detector_counter))
        detector_X.append(self.ops.append_noisy_m(circuit, loc2, N_prev, n_now, p, detector_counter))
        
        # Teleported state is in loc3
        self.ops. append_swap(circuit, loc1, loc3, N_prev, n_now)
        
        if N_prev == 1:
            return detector_0prep, detector_Z, detector_X
        else:
            return detector_0prep, detector_0prep_l2, detector_Z, detector_X


# =============================================================================
# Decoder
# ┌─────────────────────────────────────────────────────────────────────────────┐
# │                            DECODER HIERARCHY                                 │
# └─────────────────────────────────────────────────────────────────────────────┘

#                     ┌─────────────────────────────────┐
#                     │          Decoder                │
#                     │         (Abstract)              │
#                     ├─────────────────────────────────┤
#                     │ Input:  ConcatenatedCode         │
#                     ├─────────────────────────────────┤
#                     │ + decode_measurement(m, type)   │
#                     │   -> int (0 or 1)               │
#                     │                                 │
#                     │ + decode_measurement_post_      │
#                     │   selection(m, type)            │
#                     │   -> int (0, 1, or -1)          │
#                     └────────────────┬────────────────┘
#                                      │
#                     ┌────────────────┴────────────────┐
#                     │                                 │
#                     ▼                                 ▼
#      ┌──────────────────────────┐      ┌──────────────────────────┐
#      │     SteaneDecoder        │      │    GenericDecoder        │
#      ├──────────────────────────┤      ├──────────────────────────┤
#      │ Hardcoded check_matrix   │      │ Uses CSSCode. Hz/Hx       │
#      │ and logical_op for       │      │ and Lz/Lx                │
#      │ exact match              │      │                          │
#      ├──────────────────────────┤      │ Works for any CSS code   │
#      │ + decode_ec_hd(x,        │      │ with standard syndrome   │
#      │     detector_X,          │      │ decoding                 │
#      │     detector_Z,          │      │                          │
#      │     corr_x_prev,         │      └──────────────────────────┘
#      │     corr_z_prev)         │
#      │   -> (corr_x, corr_z,    │
#      │       corr_x_next,       │
#      │       corr_z_next)       │
#      │                          │
#      │ + decode_m_hd(x,         │
#      │     detector_m,          │
#      │     correction_l1)       │
#      │   -> int                 │
#      └──────────────────────────┘

# Decoding Algorithm (Steane):
# ┌─────────────────────────────────────────────────────────────────┐
# │ 1. Compute syndrome from measurement                            │
# │ 2. Lookup correction from syndrome                              │
# │ 3. Apply correction to logical outcome                          │
# │                                                                  │
# │ For Level-2 (decode_ec_hd):                                      │
# │ 1. Decode inner measurements                                     │
# │ 2. Apply propagation corrections using PropagationTables         │
# │ 3. Combine with previous corrections                             │
# │ 4. Return updated corrections for next round                     │
# └─────────────────────────────────────────────────────────────────┘
# =============================================================================

class Decoder(ABC):
    """Abstract base class for decoders."""
    
    def __init__(self, concat_code: ConcatenatedCode):
        self.concat_code = concat_code
    
    @abstractmethod
    def decode_measurement(self, m: np.ndarray, m_type: str = 'x') -> int:
        pass
    
    @abstractmethod
    def decode_measurement_post_selection(self, m: np.ndarray, m_type: str = 'x') -> int:
        pass


# Note: SteaneDecoder has been moved to concatenated_css_v10_steane.py

class GenericDecoder(Decoder):
    """
    Generic decoder that works for any CSS code. 
    
    Provides syndrome-based decoding with lookup tables, hierarchical
    decoding for concatenated codes, and propagation-aware correction.
    
    Features:
    - Auto-generates syndrome lookup tables from code specification
    - Supports minimum-weight decoding for small codes
    - Hierarchical decoding for concatenated codes
    - Propagation correction using PropagationTables
    - Post-selection support
    """
    
    def __init__(self, concat_code:  ConcatenatedCode):
        super().__init__(concat_code)
        self.code = concat_code. code_at_level(0)
        self.n = self.code.n
        
        # Build syndrome lookup tables for X and Z errors
        self._syndrome_to_error_x = self._build_syndrome_table(self.code.Hz, 'x')
        self._syndrome_to_error_z = self._build_syndrome_table(self.code. Hx, 'z')
        
        # Cache logical operators
        self._logical_x = self.code. Lx
        self._logical_z = self.code.Lz
        
        # Precompute syndrome weights for tie-breaking
        self._error_weights_x = self._compute_error_weights(self.code.Hz)
        self._error_weights_z = self._compute_error_weights(self.code.Hx)
    
    def _build_syndrome_table(self, check_matrix: np.ndarray, 
                               error_type: str) -> Dict[int, int]:
        """
        Build syndrome -> error lookup table.
        
        For each possible syndrome, find the minimum weight error that
        produces it. For distance-3 codes, this is optimal for single errors.
        
        Args:
            check_matrix: Stabilizer check matrix (Hz for X errors, Hx for Z errors)
            error_type: 'x' or 'z'
        
        Returns:
            Dict mapping syndrome (as int) to error position (or -1 for no error)
        """
        num_stabilizers = check_matrix.shape[0]
        n = check_matrix.shape[1]
        
        syndrome_table = {}
        
        # Syndrome 0 -> no error
        syndrome_table[0] = -1
        
        # Single-qubit errors
        for qubit in range(n):
            syndrome = 0
            for stab_idx in range(num_stabilizers):
                if check_matrix[stab_idx, qubit] == 1:
                    syndrome += (1 << stab_idx)
            
            # Only store if this syndrome not yet seen (first = lowest weight)
            if syndrome not in syndrome_table:
                syndrome_table[syndrome] = qubit
        
        # For higher distance codes, also consider two-qubit errors
        # This helps with distance-5+ codes
        if self.code. d >= 5:
            for q1 in range(n):
                for q2 in range(q1 + 1, n):
                    syndrome = 0
                    for stab_idx in range(num_stabilizers):
                        bit = (check_matrix[stab_idx, q1] + check_matrix[stab_idx, q2]) % 2
                        if bit == 1:
                            syndrome += (1 << stab_idx)
                    
                    # Store two-qubit error as negative composite
                    # (We'll decode this specially)
                    if syndrome not in syndrome_table:
                        syndrome_table[syndrome] = -(q1 * n + q2 + 1)  # Encode pair
        
        return syndrome_table
    
    def _compute_error_weights(self, check_matrix: np.ndarray) -> Dict[int, int]:
        """
        Compute the weight of the minimum error for each syndrome.
        
        Used for post-selection decisions. 
        """
        num_stabilizers = check_matrix. shape[0]
        n = check_matrix.shape[1]
        
        weights = {0: 0}
        
        # Single errors have weight 1
        for qubit in range(n):
            syndrome = 0
            for stab_idx in range(num_stabilizers):
                if check_matrix[stab_idx, qubit] == 1:
                    syndrome += (1 << stab_idx)
            if syndrome not in weights:
                weights[syndrome] = 1
        
        # Two-qubit errors have weight 2
        for q1 in range(n):
            for q2 in range(q1 + 1, n):
                syndrome = 0
                for stab_idx in range(num_stabilizers):
                    bit = (check_matrix[stab_idx, q1] + check_matrix[stab_idx, q2]) % 2
                    if bit == 1:
                        syndrome += (1 << stab_idx)
                if syndrome not in weights:
                    weights[syndrome] = 2
        
        return weights
    
    def _compute_syndrome(self, m: np.ndarray, check_matrix: np.ndarray) -> int:
        """
        Compute syndrome from measurement outcomes.
        
        Args:
            m: Measurement outcomes (length n)
            check_matrix:  Stabilizer check matrix
        
        Returns:
            Syndrome as integer (bit-packed)
        """
        syndrome = 0
        for stab_idx in range(check_matrix.shape[0]):
            parity = int(np.sum(m * check_matrix[stab_idx, :]) % 2)
            syndrome += parity * (1 << stab_idx)
        return syndrome
    
    def _compute_logical_value(self, m: np.ndarray, logical_op: np.ndarray) -> int:
        """Compute logical measurement value."""
        return int(np.sum(m * logical_op) % 2)
    
    def _get_correction_for_syndrome(self, syndrome: int, error_type: str) -> Tuple[int, int]:
        """
        Get error correction for a syndrome.
        
        Returns:
            (error_position, logical_flip)
            error_position: -1 if no error, qubit index otherwise
            logical_flip: 1 if correction flips logical, 0 otherwise
        """
        if error_type == 'x':
            syndrome_table = self._syndrome_to_error_x
            logical_op = self._logical_x
        else:
            syndrome_table = self._syndrome_to_error_z
            logical_op = self._logical_z
        
        if syndrome == 0:
            return -1, 0
        
        error_pos = syndrome_table. get(syndrome, None)
        
        if error_pos is None: 
            # Unknown syndrome - likely multi-qubit error beyond correction capability
            # Return no correction (will likely cause logical error)
            return -1, 0
        
        if error_pos >= 0:
            # Single qubit error
            logical_flip = int(logical_op[error_pos])
        elif error_pos < -1:
            # Two-qubit error (encoded as negative)
            # Decode the pair
            pair_code = -(error_pos + 1)
            q1 = pair_code // self.n
            q2 = pair_code % self.n
            logical_flip = int((logical_op[q1] + logical_op[q2]) % 2)
        else:
            logical_flip = 0
        
        return error_pos, logical_flip
    
    def decode_measurement(self, m: np.ndarray, m_type: str = 'x') -> int:
        """
        Decode a measurement to get logical outcome.
        
        Uses syndrome decoding to correct errors and extract logical value.
        
        Args:
            m: Measurement outcomes (length n)
            m_type: 'x' for X-basis measurement, 'z' for Z-basis
        
        Returns:
            Logical measurement outcome (0 or 1)
        """
        if m_type == 'x': 
            check_matrix = self.code. Hx
            logical_op = self._logical_x
            syndrome_table = self._syndrome_to_error_x
        else: 
            check_matrix = self.code.Hz
            logical_op = self._logical_z
            syndrome_table = self._syndrome_to_error_z
        
        # Compute raw logical value
        outcome = self._compute_logical_value(m, logical_op)
        
        # Compute syndrome
        syndrome = self._compute_syndrome(m, check_matrix)
        
        # Apply correction
        if syndrome > 0:
            error_pos = syndrome_table. get(syndrome, None)
            
            if error_pos is not None and error_pos >= 0:
                # Single qubit correction
                outcome = (outcome + int(logical_op[error_pos])) % 2
            elif error_pos is not None and error_pos < -1:
                # Two-qubit correction
                pair_code = -(error_pos + 1)
                q1 = pair_code // self. n
                q2 = pair_code % self.n
                correction = (int(logical_op[q1]) + int(logical_op[q2])) % 2
                outcome = (outcome + correction) % 2
        
        return int(outcome)
    
    def decode_measurement_post_selection(self, m: np.ndarray, m_type: str = 'x') -> int:
        """
        Decode with post-selection on high-weight syndromes.
        
        Returns -1 if syndrome indicates likely uncorrectable error.
        
        Args:
            m:  Measurement outcomes
            m_type: 'x' or 'z'
        
        Returns:
            0 or 1 for valid decode, -1 for rejected (post-selected out)
        """
        if m_type == 'x': 
            check_matrix = self.code.Hx
            logical_op = self._logical_x
            syndrome_table = self._syndrome_to_error_x
            weights = self._error_weights_x
        else:
            check_matrix = self.code.Hz
            logical_op = self._logical_z
            syndrome_table = self._syndrome_to_error_z
            weights = self._error_weights_z
        
        # Compute syndrome
        syndrome = self._compute_syndrome(m, check_matrix)
        
        # Check if syndrome is correctable
        if syndrome > 0:
            if syndrome not in syndrome_table:
                # Unknown syndrome - reject
                return -1
            
            # For distance-3 codes, reject weight-2+ errors
            if self.code. d <= 3 and weights. get(syndrome, 99) >= 2:
                return -1
        
        # Decode normally
        return self.decode_measurement(m, m_type)
    
    def decode_block(self, sample: np.ndarray, detector_info: List, 
                     level:  int, m_type: str = 'x') -> int:
        """
        Decode a single block measurement from detector info.
        
        Args:
            sample: Full detector sample array
            detector_info: [start, end] indices into sample
            level:  Concatenation level
            m_type:  Measurement basis
        
        Returns:
            Decoded logical value
        """
        if isinstance(detector_info, list) and len(detector_info) == 2:
            if isinstance(detector_info[0], int):
                m = sample[detector_info[0]: detector_info[1]]
                return self.decode_measurement(m, m_type)
        return 0
    
    def decode_hierarchical(self, sample: np. ndarray, detector_info,
                           level: int, corrections: np.ndarray = None,
                           m_type: str = 'x') -> int:
        """
        Hierarchical decoding through concatenation levels.
        
        Recursively decodes inner blocks, then decodes outer code
        treating inner outcomes as physical measurements.
        
        Args:
            sample: Full detector sample array
            detector_info:  Hierarchical detector structure
            level: Current concatenation level
            corrections:  Corrections to apply from previous EC rounds
            m_type:  Measurement basis
        
        Returns: 
            Decoded logical value at this level
        """
        code = self.concat_code. code_at_level(level)
        
        # Base case: physical level
        if level == 0:
            return self.decode_block(sample, detector_info, level, m_type)
        
        # Recursive case: decode inner blocks
        inner_outcomes = []
        for i in range(code.n):
            if isinstance(detector_info, list) and i < len(detector_info):
                inner_outcome = self.decode_hierarchical(
                    sample, detector_info[i], level - 1, None, m_type
                )
            else:
                inner_outcome = 0
            
            # Apply corrections if provided
            if corrections is not None and i < len(corrections):
                inner_outcome = (inner_outcome + int(corrections[i])) % 2
            
            inner_outcomes.append(inner_outcome)
        
        # Decode outer code using inner outcomes as measurements
        return self.decode_measurement(np.array(inner_outcomes), m_type)
    
    def decode_ec_hd(self, x:  np.ndarray, detector_X: List, detector_Z: List,
                     correction_x_prev: List, correction_z_prev: List) -> Tuple: 
        """
        Hierarchical EC decoding with propagation corrections.
        
        This is the generic version of Steane's decode_ec_hd.  It uses
        the code's propagation tables if available, otherwise falls back
        to simplified decoding.
        
        Args:
            x: Detector sample array
            detector_X: X syndrome detector info (hierarchical)
            detector_Z:  Z syndrome detector info (hierarchical)
            correction_x_prev:  X corrections from previous round
            correction_z_prev:  Z corrections from previous round
        
        Returns:
            (correction_x, correction_z, correction_x_next, correction_z_next)
        """
        n = self.n
        prop = self.concat_code.get_propagation_tables(1)
        
        # Initialize correction arrays
        mx = [0] * n
        mz = [0] * n
        correction_x_next = [0] * n
        correction_z_next = [0] * n
        
        # Handle numpy arrays and None values properly
        if correction_x_prev is None or (hasattr(correction_x_prev, '__len__') and len(correction_x_prev) == 0):
            cx1 = [0] * n
        else:
            cx1 = list(correction_x_prev)
        
        if correction_z_prev is None or (hasattr(correction_z_prev, '__len__') and len(correction_z_prev) == 0):
            cz1 = [0] * n
        else:
            cz1 = list(correction_z_prev)
        
        cx2, cz2, cx3, cz3 = [0] * n, [0] * n, [0] * n, [0] * n
        
        if prop is None:
            # Simplified decoding without propagation tables
            return self._decode_ec_simple(x, detector_X, detector_Z, cx1, cz1)
        
        num_ec = prop.num_ec_0prep
        
        # Initialize 0prep correction arrays
        cx2_0prep = [0] * num_ec
        cz2_0prep = [0] * num_ec
        cx3_0prep = [0] * num_ec
        cz3_0prep = [0] * num_ec
        
        # Decode main EC round measurements
        for i in range(n):
            idx_2 = 2 * num_ec + i
            idx_3 = 2 * num_ec + n + i
            
            if idx_2 < len(detector_X) and detector_X[idx_2]: 
                det_x = detector_X[idx_2]
                det_z = detector_Z[idx_2]
                cx2[i] = self._safe_decode_detector(x, det_x)
                cz2[i] = self._safe_decode_detector(x, det_z)
            
            if idx_3 < len(detector_X) and detector_X[idx_3]:
                det_x = detector_X[idx_3]
                det_z = detector_Z[idx_3]
                cx3[i] = self._safe_decode_detector(x, det_x)
                cz3[i] = self._safe_decode_detector(x, det_z)
        
        # Decode 0prep measurements and apply propagation
        for a in range(num_ec):
            if a < len(detector_X) and detector_X[a]:
                cx2_0prep[a] = self._safe_decode_detector(x, detector_X[a])
                cz2_0prep[a] = self._safe_decode_detector(x, detector_Z[a])
            
            if num_ec + a < len(detector_X) and detector_X[num_ec + a]:
                cx3_0prep[a] = self._safe_decode_detector(x, detector_X[num_ec + a])
                cz3_0prep[a] = self._safe_decode_detector(x, detector_Z[num_ec + a])
            
            # Apply X error propagation
            if a < len(prop.propagation_X):
                for i in prop.propagation_X[a]: 
                    if i < n: 
                        cz2[i] = (cz2[i] + cx2_0prep[a]) % 2
                        cx3[i] = (cx3[i] + cx3_0prep[a]) % 2
            
            # Apply Z error propagation
            if a < len(prop.propagation_Z):
                for i in prop. propagation_Z[a]: 
                    if i < n: 
                        cx2[i] = (cx2[i] + cz2_0prep[a]) % 2
                        cx3[i] = (cx3[i] + cz2_0prep[a]) % 2
                        cz2[i] = (cz2[i] + cz3_0prep[a]) % 2
                        cz3[i] = (cz3[i] + cz3_0prep[a]) % 2
        
        # Compute final corrections
        # Determine structure: with prep EC the structure is:
        #   [num_ec prep EC anc1] + [num_ec prep EC anc2] + [n L1 EC anc1] + [n L1 EC anc2] + [trans meas]
        #   Final index = 2*num_ec + 2*n
        # Without prep EC (corrected prep), the structure is shorter:
        #   [n L1 EC anc1] + [n L1 EC anc2] + [trans meas]
        #   Final index = 2*n
        # Detect structure based on list length vs expected
        expected_with_prep_ec = 2 * num_ec + 2 * n + 1
        expected_without_prep_ec = 2 * n + 1
        
        if len(detector_X) >= expected_with_prep_ec:
            # Structure includes prep EC
            final_idx = 2 * num_ec + 2 * n
        elif len(detector_X) >= expected_without_prep_ec:
            # No prep EC (corrected prep style)
            final_idx = 2 * n
        else:
            # Fallback: use last index (simplified decoding handles edge cases)
            final_idx = len(detector_X) - 1 if len(detector_X) > 0 else 0
        
        for i in range(n):
            x_correction = (cx1[i] + cx2[i]) % 2
            z_correction = (cz1[i] + cz2[i]) % 2
            correction_x_next[i] = cx3[i]
            correction_z_next[i] = cz3[i]
            
            # Decode final measurements with corrections
            if final_idx < len(detector_X) and i < len(detector_X[final_idx]):
                det_x = detector_X[final_idx][i]
                det_z = detector_Z[final_idx][i]
                
                if isinstance(det_x, list) and len(det_x) == 2:
                    mx[i] = (self.decode_measurement(x[det_x[0]: det_x[1]], 'x') + x_correction) % 2
                    mz[i] = (self.decode_measurement(x[det_z[0]:det_z[1]], 'z') + z_correction) % 2
        
        # Decode outer corrections
        correction_x = self. decode_measurement(np.array(mx), 'x')
        correction_z = self.decode_measurement(np.array(mz), 'z')
        
        return correction_x, correction_z, correction_x_next, correction_z_next
    
    def _decode_ec_simple(self, x: np.ndarray, detector_X: List, detector_Z: List,
                          cx_prev: List, cz_prev: List) -> Tuple:
        """
        Simplified EC decoding without propagation tables.
        
        Just decodes syndrome measurements directly without tracking
        error propagation through preparation circuit.
        """
        n = self.n
        cx_next = [0] * n
        cz_next = [0] * n
        mx = [0] * n
        mz = [0] * n
        
        # Decode the most recent syndrome measurements
        if detector_X and len(detector_X) > 0:
            last_idx = len(detector_X) - 1
            
            for i in range(min(n, len(detector_X[last_idx]) if isinstance(detector_X[last_idx], list) else 0)):
                det_x = detector_X[last_idx][i] if isinstance(detector_X[last_idx], list) else None
                det_z = detector_Z[last_idx][i] if isinstance(detector_Z[last_idx], list) else None
                
                if det_x and isinstance(det_x, list) and len(det_x) == 2:
                    mx[i] = (self.decode_measurement(x[det_x[0]:det_x[1]], 'x') + cx_prev[i]) % 2
                if det_z and isinstance(det_z, list) and len(det_z) == 2:
                    mz[i] = (self.decode_measurement(x[det_z[0]:det_z[1]], 'z') + cz_prev[i]) % 2
        
        correction_x = self.decode_measurement(np.array(mx), 'x')
        correction_z = self.decode_measurement(np.array(mz), 'z')
        
        return correction_x, correction_z, cx_next, cz_next
    
    def _safe_decode_detector(self, x: np.ndarray, detector_info) -> int:
        """
        Safely decode a detector, handling various formats.
        """
        if detector_info is None:
            return 0
        
        if isinstance(detector_info, list):
            if len(detector_info) == 0:
                return 0
            
            # [start, end] format
            if len(detector_info) == 2 and isinstance(detector_info[0], int):
                return self.decode_measurement(x[detector_info[0]:detector_info[1]])
            
            # Nested format - take first element
            if isinstance(detector_info[0], list):
                return self._safe_decode_detector(x, detector_info[0])
        
        return 0
    
    def decode_m_hd(self, x: np. ndarray, detector_m: List, 
                    correction_l1: List) -> int:
        """
        Hierarchical measurement decoding with corrections.
        
        Decodes each inner block measurement and applies corrections,
        then decodes the outer code.
        
        Args:
            x: Detector sample array
            detector_m: Measurement detector info (hierarchical)
            correction_l1: Corrections to apply at level-1
        
        Returns: 
            Decoded logical measurement outcome
        """
        n = self.n
        m = [0] * n
        
        for i in range(n):
            if i < len(detector_m):
                det = detector_m[i]
                
                if isinstance(det, list) and len(det) == 2 and isinstance(det[0], int):
                    # Direct [start, end] format
                    raw = self.decode_measurement(x[det[0]:det[1]])
                else:
                    # Hierarchical - decode recursively
                    raw = self._safe_decode_detector(x, det)
                
                correction = int(correction_l1[i]) if i < len(correction_l1) else 0
                m[i] = (raw + correction) % 2
        
        return self.decode_measurement(np.array(m), 'x')
    
    def compute_threshold_estimate(self) -> float:
        """
        Estimate the error threshold for this code.
        
        Uses the code distance and structure to estimate where
        logical error rate crosses physical error rate.
        
        Returns:
            Estimated threshold probability
        """
        d = self.code.d
        n = self.code.n
        
        # Rough estimate:  threshold ~ 1 / (c * n) where c depends on structure
        # For CSS codes, threshold is typically in range 0.1% - 1%
        
        # Better estimate using distance
        # p_L ~ (p / p_th)^((d+1)/2) for concatenated codes
        # At threshold, p_L = p, so p_th ~ O(1/n)
        
        c = 10  # Empirical constant
        threshold = 1.0 / (c * n)
        
        return threshold
    
    def get_code_info(self) -> Dict:
        """
        Get information about the code being decoded.
        """
        return {
            'name': self.code.name,
            'n': self.n,
            'k': self.code.k,
            'd': self.code.d,
            'num_x_stabilizers': self.code.num_x_stabilizers,
            'num_z_stabilizers': self.code.num_z_stabilizers,
            'syndrome_table_size_x': len(self._syndrome_to_error_x),
            'syndrome_table_size_z': len(self._syndrome_to_error_z),
        }

# =============================================================================
# Post-Selection
# ┌─────────────────────────────────────────────────────────────────────────────┐
# │                     POST-SELECTION & ACCEPTANCE                              │
# └─────────────────────────────────────────────────────────────────────────────┘

# ┌─────────────────────────────────┐      ┌─────────────────────────────────┐
# │        PostSelector             │      │      AcceptanceChecker          │
# ├─────────────────────────────────┤      ├─────────────────────────────────┤
# │ Input: ConcatenatedCode,        │      │ Input: ConcatenatedCode,        │
# │        Decoder                  │      │        Decoder                  │
# ├─────────────────────────────────┤      ├─────────────────────────────────┤
# │ + post_selection_steane(x,      │      │ + accept_l1(x, detector_m,      │
# │     detector_0prep) -> bool     │      │     detector_X, detector_Z, Q)  │
# │                                 │      │   -> float (error probability)  │
# │ + post_selection_steane_l2(x,   │      │                                 │
# │     detector_0prep,             │      │ + accept_l2(x, detector_m,      │
# │     detector_X, detector_Z)     │      │     detector_X, detector_Z, Q)  │
# │   -> bool                       │      │   -> float                      │
# │                                 │      │                                 │
# │ + post_selection_l1(x,          │      │ Uses decoder to compute         │
# │     list_detector_0prep)        │      │ corrections and check if        │
# │   -> bool                       │      │ Bell pair correlations hold     │
# │                                 │      │                                 │
# │ + post_selection_l2(x, ...)     │      │ Returns 0 if no error,          │
# │   -> bool                       │      │ 1 if definite error,            │
# │                                 │      │ 0.5 if uncertain                │
# └─────────────────────────────────┘      └─────────────────────────────────┘
# =============================================================================

class PostSelector:
    """Post-selection matching original."""
    
    def __init__(self, concat_code:  ConcatenatedCode, decoder: Decoder):
        self.concat_code = concat_code
        self.decoder = decoder
    
    def post_selection_steane(self, x: np.ndarray, detector_0prep: List) -> bool:
        """Level-1 post-selection on single detector."""
        if x[detector_0prep[0]] % 2 == 0:
            return True
        return False
    
    def post_selection_steane_l2(self, x: np.ndarray, detector_0prep: List,
                                  detector_X: List, detector_Z: List) -> bool:
        """Level-2 post-selection with propagation."""
        prop = self.concat_code.get_propagation_tables(1)
        if prop is None: 
            return True
        
        outcome = self.decoder.decode_measurement(
            x[detector_0prep[0]:detector_0prep[1]]
        )
        
        for a in prop.propagation_m:
            if a < len(detector_X) and detector_X[a]: 
                correction_x = self.decoder.decode_measurement(
                    x[detector_X[a][0][0]:detector_X[a][0][1]]
                )
                outcome = (outcome + correction_x) % 2
        
        return outcome % 2 == 0
    
    def post_selection_l1(self, x: np.ndarray, list_detector_0prep: List) -> bool:
        """Full level-1 post-selection."""
        for a in list_detector_0prep:
            if len(a) == 1:
                if not self.post_selection_steane(x, a[0]):
                    return False
            elif len(a) == 2:
                if a[1] - a[0] == 1:
                    if not self.post_selection_steane(x, a):
                        return False
        return True
    
    def post_selection_l2(self, x: np.ndarray, list_detector_0prep: List,
                          list_detector_0prep_l2: List, list_detector_X: List,
                          list_detector_Z: List, Q: int) -> bool:
        """Full level-2 post-selection."""
        prop = self.concat_code.get_propagation_tables(1)
        if prop is None: 
            return True
        
        num_ec = prop.num_ec_0prep
        num_correction = 2 * Q
        
        for a in list_detector_0prep: 
            if len(a) == 1:
                if not self.post_selection_steane(x, a[0]):
                    return False
            elif len(a) == 2:
                if a[1] - a[0] == 1:
                    if not self.post_selection_steane(x, a):
                        return False
        
        for i in range(num_correction):
            if 2 * i < len(list_detector_0prep_l2):
                if not self.post_selection_steane_l2(
                    x, list_detector_0prep_l2[2 * i],
                    list_detector_X[i][0:num_ec],
                    list_detector_Z[i][0:num_ec]
                ):
                    return False
            if 2 * i + 1 < len(list_detector_0prep_l2):
                if not self.post_selection_steane_l2(
                    x, list_detector_0prep_l2[2 * i + 1],
                    list_detector_X[i][num_ec:2 * num_ec],
                    list_detector_Z[i][num_ec:2 * num_ec]
                ):
                    return False
        
        return True
    
    def post_selection_l2_memory(self, x: np.ndarray, list_detector_0prep: List,
                                  list_detector_0prep_l2: List, list_detector_X: List,
                                  list_detector_Z: List, Q: int,
                                  detector_X_prep: List = None,
                                  detector_Z_prep: List = None) -> bool:
        """
        Post-selection for single-qubit L2 memory experiment.
        Unlike post_selection_l2 which expects 2 logical qubits (for CNOT),
        this function handles the case of 1 logical qubit with Q rounds of EC.
        
        Args:
            x: Sample vector
            list_detector_0prep: Level-1 preparation detectors
            list_detector_0prep_l2: Level-2 verification detectors [prep_l2, ec_anc1_l2, ec_anc2_l2, ...]
            list_detector_X: EC round X detectors
            list_detector_Z: EC round Z detectors
            Q: Number of EC rounds
            detector_X_prep: X detectors from noisy prep (for prep L2 verification)
            detector_Z_prep: Z detectors from noisy prep (for prep L2 verification)
        """
        prop = self.concat_code.get_propagation_tables(1)
        if prop is None:
            return True
        
        num_ec = prop.num_ec_0prep
        
        # Post-select on level-1 preparation detectors
        for a in list_detector_0prep:
            if len(a) == 1:
                if not self.post_selection_steane(x, a[0]):
                    return False
            elif len(a) == 2:
                if a[1] - a[0] == 1:
                    if not self.post_selection_steane(x, a):
                        return False
        
        # Post-select on level-2 preparation verification (index 0)
        # This is CRITICAL for fault tolerance - verifies the prep was successful
        # NOTE: With corrected prep (v2 style), detector_X_prep and detector_Z_prep are empty 
        # (no EC during prep). In that case, we only verify the L2 verification measurement.
        if len(list_detector_0prep_l2) > 0 and list_detector_0prep_l2[0] is not None:
            # Check if we have prep EC detectors (old-style prep) or not (corrected prep)
            if (detector_X_prep is not None and len(detector_X_prep) > 0 
                and detector_Z_prep is not None and len(detector_Z_prep) > 0):
                if not self.post_selection_steane_l2(
                    x, list_detector_0prep_l2[0],
                    detector_X_prep[0:num_ec],
                    detector_Z_prep[0:num_ec]
                ):
                    return False
            else:
                # Corrected prep: just verify the L2 measurement is clean (all zeros after decoding)
                detector_l2 = list_detector_0prep_l2[0]
                if isinstance(detector_l2, list) and len(detector_l2) == 2:
                    if self.decoder.decode_measurement(x[detector_l2[0]:detector_l2[1]]) != 0:
                        return False
        
        # Check EC round l2 detectors
        # NOTE: With corrected prep (v2), the EC round's detector_X/detector_Z structure is different:
        # - Old (with prep EC): [num_ec prep EC anc1] + [num_ec prep EC anc2] + [n L1 EC anc1] + [n L1 EC anc2] + [trans meas]
        # - New (no prep EC): [n L1 EC anc1] + [n L1 EC anc2] + [trans meas]
        # Detect based on list length vs expected structure:
        code = self.concat_code.code_at_level(0)
        n = code.n
        expected_with_prep_ec = 2 * num_ec + 2 * n + 1  # Structure with prep EC
        expected_without_prep_ec = 2 * n + 1  # Structure without prep EC
        
        for i in range(Q):
            # Each EC round adds 2 entries to list_detector_0prep_l2 (one per ancilla block)
            # Index into list_detector_0prep_l2: 1 + 2*i and 1 + 2*i + 1 (after prep at index 0)
            idx_base = 1 + 2 * i
            
            # Check if we have old structure (with prep EC) or new structure (no prep EC)
            # using proper structure comparison instead of magic number heuristic
            actual_len = len(list_detector_X[i]) if i < len(list_detector_X) else 0
            has_prep_ec = actual_len >= expected_with_prep_ec
            
            if has_prep_ec:
                # Old-style: use num_ec_0prep indexing
                if idx_base < len(list_detector_0prep_l2) and i < len(list_detector_X):
                    if not self.post_selection_steane_l2(
                        x, list_detector_0prep_l2[idx_base],
                        list_detector_X[i][0:num_ec],
                        list_detector_Z[i][0:num_ec]
                    ):
                        return False
                if idx_base + 1 < len(list_detector_0prep_l2) and i < len(list_detector_X):
                    if not self.post_selection_steane_l2(
                        x, list_detector_0prep_l2[idx_base + 1],
                        list_detector_X[i][num_ec:2 * num_ec],
                        list_detector_Z[i][num_ec:2 * num_ec]
                    ):
                        return False
            else:
                # New-style (v2 prep): just check L2 verification measurements
                # With no prep EC, we can only verify the L2 measurement itself
                if idx_base < len(list_detector_0prep_l2) and list_detector_0prep_l2[idx_base] is not None:
                    detector_l2 = list_detector_0prep_l2[idx_base]
                    if isinstance(detector_l2, list) and len(detector_l2) == 2:
                        if self.decoder.decode_measurement(x[detector_l2[0]:detector_l2[1]]) != 0:
                            return False
                if idx_base + 1 < len(list_detector_0prep_l2) and list_detector_0prep_l2[idx_base + 1] is not None:
                    detector_l2 = list_detector_0prep_l2[idx_base + 1]
                    if isinstance(detector_l2, list) and len(detector_l2) == 2:
                        if self.decoder.decode_measurement(x[detector_l2[0]:detector_l2[1]]) != 0:
                            return False
        
        return True


# =============================================================================
# Acceptance Functions
# =============================================================================

class AcceptanceChecker:
    """Acceptance/error checking matching original."""
    
    def __init__(self, concat_code:  ConcatenatedCode, decoder: Decoder):
        self.concat_code = concat_code
        self.decoder = decoder
    
    def accept_l1(self, x: np. ndarray, list_detector_m: List,
                  list_detector_X: List, list_detector_Z: List, Q: int) -> float:
        """Level-1 acceptance check matching original."""
        num_correction = 2 * Q
        X_propagate = [[1], [3]]
        Z_propagate = [[0], [2]]
        outcome = np.zeros(4)
        correction_x = np.zeros(num_correction)
        correction_z = np.zeros(num_correction)
        
        for i in range(num_correction):
            correction_x[i] = self. decoder.decode_measurement(
                x[list_detector_X[i][0][0]:list_detector_X[i][0][1]]
            )
            correction_z[i] = self. decoder.decode_measurement(
                x[list_detector_Z[i][0][0]:list_detector_Z[i][0][1]]
            )
        
        for i in range(4):
            outcome[i] = self.decoder.decode_measurement(
                x[list_detector_m[i][0]: list_detector_m[i][1]]
            )
        
        for i in range(num_correction):
            pos = i % 2
            for x_prop in X_propagate[pos]:
                if outcome[x_prop] == -1:
                    continue
                if correction_x[i] == 1:
                    outcome[x_prop] = (outcome[x_prop] + 1) % 2
                if correction_x[i] == -1:
                    outcome[x_prop] = -1
            for z_prop in Z_propagate[pos]:
                if outcome[z_prop] == -1:
                    continue
                if correction_z[i] == 1:
                    outcome[z_prop] = (outcome[z_prop] + 1) % 2
                if correction_z[i] == -1:
                    outcome[z_prop] = -1
        
        flag = 1
        for i in range(4):
            if outcome[i] == 1:
                flag = 0
            if outcome[i] == -1:
                flag *= 0.5
        
        return 1 - flag
    
    def accept_l2(self, x: np.ndarray, list_detector_m: List,
                  list_detector_X: List, list_detector_Z: List, Q: int) -> float:
        """Level-2 acceptance check matching original."""
        # Get inner code size dynamically
        inner_n = self.concat_code.code_at_level(0).n
        
        num_correction = 2 * Q
        X_propagate = [[1], [3]]
        Z_propagate = [[0], [2]]
        outcome = np.zeros(4)
        correction_x = np.zeros(num_correction)
        correction_z = np.zeros(num_correction)
        correction1_x = np.zeros(inner_n)
        correction1_z = np.zeros(inner_n)
        correction2_x = np.zeros(inner_n)
        correction2_z = np.zeros(inner_n)
        
        for i in range(Q):
            correction_x[2*i], correction_z[2*i], correction1_x, correction1_z = \
                self.decoder.decode_ec_hd(x, list_detector_X[2*i], list_detector_Z[2*i],
                                          correction1_x, correction1_z)
            correction_x[2*i+1], correction_z[2*i+1], correction2_x, correction2_z = \
                self.decoder.decode_ec_hd(x, list_detector_X[2*i+1], list_detector_Z[2*i+1],
                                          correction2_x, correction2_z)
        
        outcome[0] = self.decoder. decode_m_hd(x, list_detector_m[0], correction1_z)
        outcome[1] = self.decoder.decode_m_hd(x, list_detector_m[1], correction1_x)
        outcome[2] = self.decoder.decode_m_hd(x, list_detector_m[2], correction2_z)
        outcome[3] = self.decoder.decode_m_hd(x, list_detector_m[3], correction2_x)
        
        for i in range(num_correction):
            pos = i % 2
            for x_prop in X_propagate[pos]:
                if outcome[x_prop] == -1:
                    continue
                if correction_x[i] == 1:
                    outcome[x_prop] = (outcome[x_prop] + 1) % 2
                if correction_x[i] == -1:
                    outcome[x_prop] = -1
            for z_prop in Z_propagate[pos]:
                if outcome[z_prop] == -1:
                    continue
                if correction_z[i] == 1:
                    outcome[z_prop] = (outcome[z_prop] + 1) % 2
                if correction_z[i] == -1:
                    outcome[z_prop] = -1
        
        flag = 1
        for i in range(4):
            if outcome[i] == 1:
                flag = 0
            if outcome[i] == -1:
                flag *= 0.5
        
        return 1 - flag


# =============================================================================
# Simulator
# ┌─────────────────────────────────────────────────────────────────────────────┐
# │                        MAIN SIMULATOR                                        │
# └─────────────────────────────────────────────────────────────────────────────┘

# ┌─────────────────────────────────────────────────────────────────────────────┐
# │                     ConcatenatedCodeSimulator                                │
# ├─────────────────────────────────────────────────────────────────────────────┤
# │ Input:                                                                        │
# │   - concat_code: ConcatenatedCode                                            │
# │   - noise_model: NoiseModel                                                  │
# │   - use_steane_strategy: bool (auto-detected if None)                        │
# ├─────────────────────────────────────────────────────────────────────────────┤
# │ Components (created in __init__):                                            │
# │                                                                              │
# │   ┌─────────────────┐                                                        │
# │   │ TransversalOps  │◄──────────────────────────────────────┐                │
# │   └────────┬────────┘                                       │                │
# │            │                                                │                │
# │            ▼                                                │                │
# │   ┌─────────────────┐    ┌─────────────────┐               │                │
# │   │   ECGadget      │◄──►│ Preparation     │               │                │
# │   │ (Steane/Knill)  │    │  Strategy       │               │                │
# │   └────────┬────────┘    └────────┬────────┘               │                │
# │            │                      │                        │                │
# │            │                      │                        │                │
# │   ┌────────┴──────────────────────┴────────┐               │                │
# │   │                                        │               │                │
# │   ▼                                        ▼               │                │
# │   ┌─────────────────┐             ┌─────────────────┐      │                │
# │   │    Decoder      │             │ PostSelector    │      │                │
# │   │ (Steane/Generic)│             │                 │      │                │
# │   └────────┬────────┘             └────────┬────────┘      │                │
# │            │                               │               │                │
# │            ▼                               │               │                │
# │   ┌─────────────────┐                      │               │                │
# │   │ Acceptance      │◄─────────────────────┘               │                │
# │   │   Checker       │                                      │                │
# │   └─────────────────┘                                      │                │
# │                                                            │                │
# │   ┌─────────────────┐                                      │                │
# │   │ LogicalGate     │──────────────────────────────────────┘                │
# │   │   Dispatcher    │                                                        │
# │   └─────────────────┘                                                        │
# ├─────────────────────────────────────────────────────────────────────────────┤
# │ Methods:                                                                     │
# │                                                                              │
# │ + estimate_logical_cnot_error_l1(p, num_shots, Q=10)                         │
# │   -> (logical_error:  float, variance: float)                                 │
# │                                                                              │
# │ + estimate_logical_cnot_error_l2(p, num_shots, Q=1)                          │
# │   -> (logical_error: float, variance: float)                                 │
# │                                                                              │
# │ + estimate_memory_logical_error_l1(p, num_shots, num_ec_rounds=1)            │
# │   -> (logical_error: float, variance:  float)                                 │
# │                                                                              │
# │ + estimate_memory_logical_error_l2(p, num_shots, num_ec_rounds=1)            │
# │   -> (logical_error: float, variance: float)                                 │
# └─────────────────────────────────────────────────────────────────────────────┘
# =============================================================================

class ConcatenatedCodeSimulator:
    """
    Main simulator for concatenated CSS codes.

    ┌─────────────────────────────────────────────────────────────────────────────┐
│                        SIMULATION DATA FLOW                                  │
└─────────────────────────────────────────────────────────────────────────────┘

                    ┌─────────────────┐
                    │  User Request   │
                    │  (p, num_shots) │
                    └────────┬────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         BUILD CIRCUIT PHASE                                  │
│                                                                              │
│  1. Prepare ideal Bell pairs (append_0prep)                                  │
│  2. Apply H and CNOT to create entanglement                                  │
│  3. For Q rounds:                                                            │
│     a. Apply ideal CNOT                                                      │
│     b. Apply noisy CNOT                                                      │
│     c. Apply EC gadget (append_noisy_ec)                                     │
│        - Collects detector_0prep, detector_X, detector_Z                     │
│  4. Undo Bell pairs                                                          │
│  5. Measure all qubits (append_m)                                            │
│        - Collects detector_m                                                 │
│                                                                              │
│  Output:  stim. Circuit + detector info                                        │
└─────────────────────────────────────────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         NOISE APPLICATION                                    │
│                                                                              │
│  noise_model. apply(circuit) -> noisy_circuit                                 │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                            SAMPLING                                          │
│                                                                              │
│  noisy_circuit.compile_detector_sampler().sample(shots=num_shots)            │
│  -> samples: np.ndarray[num_shots, num_detectors]                            │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         POST-SELECTION                                       │
│                                                                              │
│  For each sample:                                                            │
│    - Check verification measurements (post_selection_l1 or _l2)              │
│    - Reject samples with non-zero verification outcomes                      │
│                                                                              │
│  Output: filtered samples                                                    │
└─────────────────────────────────────────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                       DECODING & ACCEPTANCE                                  │
│                                                                              │
│  For each accepted sample:                                                   │
│    L1: decode_measurement on detector_X, detector_Z                          │
│    L2: decode_ec_hd with propagation corrections                             │
│                                                                              │
│  Check Bell pair correlations:                                                │
│    - All four measurements should decode to 0                                │
│    - Count errors when they don't                                            │
│                                                                              │
│  Output: num_accepted, num_errors                                            │
└─────────────────────────────────────────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         COMPUTE RESULTS                                      │
│                                                                              │
│  logical_error = num_errors / (num_accepted * Q)                             │
│  variance = num_errors / (num_accepted * Q)^2                                │
│                                                                              │
│  Output: (logical_error, variance)                                           │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                         EXTENSION POINTS                                     │
└─────────────────────────────────────────────────────────────────────────────┘

To add a new CSS code: 
┌─────────────────────────────────────────────────────────────────────────────┐
│ 1. Create CSSCode with:                                                       │
│    - Hz, Hx, Lz, Lx matrices                                                 │
│    - h_qubits, encoding_cnots, verification_qubits                           │
│                                                                              │
│ 2. (Optional) Create PropagationTables if using level-2+                     │
│                                                                              │
│ 3. (Optional) Create custom PreparationStrategy if needed                    │
│                                                                              │
│ 4. Use GenericPreparationStrategy and GenericDecoder for basic support       │
└─────────────────────────────────────────────────────────────────────────────┘

To add a new gate implementation:
┌─────────────────────────────────────────────────────────────────────────────┐
│ 1. Subclass LogicalHGate, LogicalCNOTGate, or LogicalMeasurement             │
│                                                                              │
│ 2. Implement apply() method                                                  │
│                                                                              │
│ 3. Register with LogicalGateDispatcher:                                       │
│    dispatcher.set_h_gate(MyCustomHGate(concat_code, ops))                    │
└─────────────────────────────────────────────────────────────────────────────┘

To add a new EC strategy:
┌─────────────────────────────────────────────────────────────────────────────┐
│ 1. Subclass ECGadget                                                         │
│                                                                              │
│ 2. Implement append_noisy_ec() with correct return structure                 │
│                                                                              │
│ 3. Wire up with PreparationStrategy                                          │
└─────────────────────────────────────────────────────────────────────────────┘
    
    Supports both Steane-specific and generic code simulations.
    """
    
    def __init__(self, concat_code: ConcatenatedCode, noise_model: NoiseModel,
                 ec_gadget: ECGadget = None,
                 prep_strategy: PreparationStrategy = None,
                 decoder: Decoder = None):
        """
        Initialize the simulator with optional custom components.
        
        Args:
            concat_code: The concatenated code to simulate
            noise_model: Noise model to apply to circuits
            ec_gadget: Custom EC gadget (optional, defaults to KnillECGadget)
            prep_strategy: Custom preparation strategy (optional, defaults to GenericPreparationStrategy)
            decoder: Custom decoder (optional, defaults to GenericDecoder)
        
        Note: For Steane-specific components, use create_steane_simulator from
        concatenated_css_v10_steane.py instead.
        """
        self.concat_code = concat_code
        self.noise_model = noise_model
        self.ops = TransversalOps(concat_code)
        
        # Create EC gadget (default to Knill, or use custom)
        if ec_gadget is not None:
            self.ec = ec_gadget
        else:
            self.ec = KnillECGadget(concat_code, self.ops)
        
        # Create preparation strategy (default to generic)
        if prep_strategy is not None:
            self.prep = prep_strategy
        else:
            self.prep = GenericPreparationStrategy(concat_code, self.ops)
        
        # Create decoder (default to generic)
        if decoder is not None:
            self.decoder = decoder
        else:
            self.decoder = GenericDecoder(concat_code)
        
        # Wire up circular dependencies
        self.ec.set_prep(self.prep)
        self.prep.set_ec_gadget(self.ec)
        
        self.post_selector = PostSelector(concat_code, self.decoder)
        self.acceptance = AcceptanceChecker(concat_code, self.decoder)
        
        # Gate dispatcher for logical operations
        self.gates = LogicalGateDispatcher(concat_code, self.ops)
    
    def estimate_logical_cnot_error_l1(self, p: float, num_shots: int,
                                        Q: int = 10) -> Tuple[float, float]:
        """
        Estimate level-1 logical CNOT error rate.
        Matches original estimate_logical_cnot_error_l1.
        """
        N_prev = 1
        N_now = self.concat_code.code_at_level(0).n
        NN = 2 * N_now
        
        list_detector_0prep = []
        list_detector_X = []
        list_detector_Z = []
        list_detector_m = []
        
        circuit = stim.Circuit()
        detector_counter = [0]
        
        # Prepare ideal Bell pairs
        self.prep.append_0prep(circuit, 0, N_prev, N_now)
        self.prep.append_0prep(circuit, NN, N_prev, N_now)
        self.prep.append_0prep(circuit, 2 * NN, N_prev, N_now)
        self.prep.append_0prep(circuit, 3 * NN, N_prev, N_now)
        
        self.ops.append_h(circuit, 0, N_prev, N_now)
        self.ops.append_h(circuit, 2 * NN, N_prev, N_now)
        
        self.ops.append_cnot(circuit, 0, NN, N_prev, N_now)
        self.ops.append_cnot(circuit, 2 * NN, 3 * NN, N_prev, N_now)
        
        # Q rounds of CNOT + EC
        for q in range(Q):
            self.ops.append_cnot(circuit, 0, 2 * NN, N_prev, N_now)
            self.ops.append_noisy_cnot(circuit, 0, 2 * NN, N_prev, N_now, p)
            
            result = self.ec.append_noisy_ec(
                circuit, 0, 4 * NN, 5 * NN, 6 * NN, N_prev, N_now, p, detector_counter
            )
            list_detector_0prep.extend(result[0])
            list_detector_X.append(result[2])
            list_detector_Z.append(result[1])
            
            result = self.ec.append_noisy_ec(
                circuit, 2 * NN, 4 * NN, 5 * NN, 6 * NN, N_prev, N_now, p, detector_counter
            )
            list_detector_0prep.extend(result[0])
            list_detector_X.append(result[2])
            list_detector_Z.append(result[1])
        
        # Undo Bell pairs
        self.ops.append_cnot(circuit, 0, NN, N_prev, N_now)
        self.ops.append_cnot(circuit, 2 * NN, 3 * NN, N_prev, N_now)
        
        self.ops.append_h(circuit, 0, N_prev, N_now)
        self.ops.append_h(circuit, 2 * NN, N_prev, N_now)
        
        # Measure
        list_detector_m. append(self.ops.append_m(circuit, 0, N_prev, N_now, detector_counter))
        list_detector_m.append(self.ops.append_m(circuit, NN, N_prev, N_now, detector_counter))
        list_detector_m.append(self.ops.append_m(circuit, 2 * NN, N_prev, N_now, detector_counter))
        list_detector_m.append(self. ops.append_m(circuit, 3 * NN, N_prev, N_now, detector_counter))
        
        # Sample (noise already applied inline by append_noisy_* functions)
        samples = circuit.compile_detector_sampler().sample(shots=num_shots)
        
        # Post-selection
        samples = [x for x in samples if self.post_selector.post_selection_l1(x, list_detector_0prep)]
        num = len(samples)
        
        # Count errors
        err = sum([self.acceptance.accept_l1(x, list_detector_m, list_detector_X,
                                              list_detector_Z, Q) for x in samples])
        
        print(p, num, err)
        
        if num > 0:
            logical_error = err / (num * Q)
            variance = err / (num * Q) ** 2
        else:
            logical_error = variance = 0
        
        return logical_error, variance
    
    def estimate_logical_cnot_error_l2(self, p: float, num_shots:  int,
                                        Q:  int = 1) -> Tuple[float, float]:
        """
        Estimate level-2 logical CNOT error rate.
        Matches original estimate_logical_cnot_error_l2.
        """
        N_prev = self.concat_code.code_at_level(0).n
        N_now = self.concat_code.code_at_level(0).n
        NN = 2 * N_now
        
        list_detector_0prep = []
        list_detector_0prep_l2 = []
        list_detector_X = []
        list_detector_Z = []
        list_detector_m = []
        
        circuit = stim.Circuit()
        detector_counter = [0]
        
        # Prepare ideal Bell pairs
        self.prep.append_0prep(circuit, 0, N_prev, N_now)
        self.prep.append_0prep(circuit, NN, N_prev, N_now)
        self.prep.append_0prep(circuit, 2 * NN, N_prev, N_now)
        self.prep.append_0prep(circuit, 3 * NN, N_prev, N_now)
        
        self.ops.append_h(circuit, 0, N_prev, N_now)
        self.ops.append_h(circuit, 2 * NN, N_prev, N_now)
        
        self.ops.append_cnot(circuit, 0, NN, N_prev, N_now)
        self.ops.append_cnot(circuit, 2 * NN, 3 * NN, N_prev, N_now)
        
        # Q rounds of CNOT + EC
        for q in range(Q):
            self.ops. append_cnot(circuit, 0, 2 * NN, N_prev, N_now)
            self.ops.append_noisy_cnot(circuit, 0, 2 * NN, N_prev, N_now, p)
            
            result = self.ec.append_noisy_ec(
                circuit, 0, 4 * NN, 5 * NN, 6 * NN, N_prev, N_now, p, detector_counter
            )
            list_detector_0prep.extend(result[0])
            list_detector_0prep_l2.extend(result[1])
            list_detector_X.append(result[3])
            list_detector_Z.append(result[2])
            
            result = self.ec.append_noisy_ec(
                circuit, 2 * NN, 4 * NN, 5 * NN, 6 * NN, N_prev, N_now, p, detector_counter
            )
            list_detector_0prep.extend(result[0])
            list_detector_0prep_l2.extend(result[1])
            list_detector_X.append(result[3])
            list_detector_Z.append(result[2])
        
        # Undo Bell pairs
        self.ops.append_cnot(circuit, 0, NN, N_prev, N_now)
        self.ops.append_cnot(circuit, 2 * NN, 3 * NN, N_prev, N_now)
        
        self.ops.append_h(circuit, 0, N_prev, N_now)
        self.ops.append_h(circuit, 2 * NN, N_prev, N_now)
        
        # Measure
        list_detector_m. append(self.ops.append_m(circuit, 0, N_prev, N_now, detector_counter))
        list_detector_m.append(self.ops.append_m(circuit, NN, N_prev, N_now, detector_counter))
        list_detector_m.append(self.ops.append_m(circuit, 2 * NN, N_prev, N_now, detector_counter))
        list_detector_m.append(self.ops.append_m(circuit, 3 * NN, N_prev, N_now, detector_counter))
        
        # Sample (noise already applied inline by append_noisy_* functions)
        samples = circuit.compile_detector_sampler().sample(shots=num_shots)
        
        # Post-selection
        samples = [x for x in samples if self.post_selector.post_selection_l2(
            x, list_detector_0prep, list_detector_0prep_l2,
            list_detector_X, list_detector_Z, Q
        )]
        num = len(samples)
        
        # Count errors
        err = sum([self.acceptance.accept_l2(x, list_detector_m, list_detector_X,
                                              list_detector_Z, Q) for x in samples])
        
        print(p, num, err)
        
        if num > 0:
            logical_error = err / (num * Q)
            variance = err / (num * Q) ** 2
        else:
            logical_error = variance = 0
        
        return logical_error, variance
    
    def estimate_memory_logical_error_l1(self, p: float, num_shots: int,
                                          num_ec_rounds: int = 1) -> Tuple[float, float]: 
        """
        Estimate level-1 memory logical error rate.
        
        Prepares |0⟩_L, applies EC rounds, measures, checks if outcome is 0.
        """
        N_prev = 1
        N_now = self.concat_code.code_at_level(0).n
        NN = 2 * N_now
        
        list_detector_0prep = []
        list_detector_X = []
        list_detector_Z = []
        
        circuit = stim.Circuit()
        detector_counter = [0]
        
        # Prepare |0⟩_L
        self.prep.append_0prep(circuit, 0, N_prev, N_now)
        
        # EC rounds
        for _ in range(num_ec_rounds):
            result = self.ec.append_noisy_ec(
                circuit, 0, NN, 2 * NN, 3 * NN, N_prev, N_now, p, detector_counter
            )
            list_detector_0prep.extend(result[0])
            list_detector_X.append(result[2])
            list_detector_Z.append(result[1])
        
        # Measure
        detector_m = self.ops.append_m(circuit, 0, N_prev, N_now, detector_counter)
        
        # Sample (noise already applied inline by append_noisy_* functions)
        samples = circuit.compile_detector_sampler().sample(shots=num_shots)
        
        # Post-selection
        samples = [x for x in samples if self.post_selector.post_selection_l1(x, list_detector_0prep)]
        num = len(samples)
        
        # Count errors (outcome should be 0 for |0⟩_L)
        num_errors = 0
        for x in samples: 
            outcome = self.decoder.decode_measurement(x[detector_m[0]:detector_m[1]])
            if outcome != 0:
                num_errors += 1
        
        print(f"Memory L1: p={p}, accepted={num}, errors={num_errors}")
        
        if num > 0:
            logical_error = num_errors / num
            variance = num_errors / (num ** 2)
        else:
            logical_error = variance = 0
        
        return logical_error, variance
    
    def estimate_memory_logical_error_l2(self, p: float, num_shots: int,
                                          num_ec_rounds: int = 1) -> Tuple[float, float]:
        """
        Estimate level-2 memory logical error rate.
        
        Prepares |0⟩_L at level 2 (noisy), applies EC rounds, measures, checks if outcome is 0.
        """
        N_prev = self.concat_code.code_at_level(0).n
        N_now = self.concat_code.code_at_level(0).n
        NN = 2 * N_now
        
        list_detector_0prep = []
        list_detector_0prep_l2 = []
        list_detector_X = []
        list_detector_Z = []
        
        circuit = stim.Circuit()
        detector_counter = [0]
        
        # Prepare |0⟩_L at level 2 (NOISY prep)
        # IMPORTANT: Save detector_X_prep and detector_Z_prep for L2 post-selection on prep
        prep_result = self.prep.append_noisy_0prep(circuit, 0, NN, N_prev, N_now, p, detector_counter)
        list_detector_0prep.extend(prep_result[0])
        list_detector_0prep_l2.append(prep_result[1])
        detector_X_prep = prep_result[2]  # X detectors from prep (for L2 verification)
        detector_Z_prep = prep_result[3]  # Z detectors from prep (for L2 verification)
        
        # EC rounds
        for _ in range(num_ec_rounds):
            result = self.ec.append_noisy_ec(
                circuit, 0, NN, 2 * NN, 3 * NN, N_prev, N_now, p, detector_counter
            )
            list_detector_0prep.extend(result[0])
            list_detector_0prep_l2.extend(result[1])
            list_detector_X.append(result[3])
            list_detector_Z.append(result[2])
        
        # Measure
        detector_m = self.ops.append_m(circuit, 0, N_prev, N_now, detector_counter)
        
        # Sample (noise already applied inline by append_noisy_* functions)
        samples = circuit.compile_detector_sampler().sample(shots=num_shots)
        
        # Post-selection with prep's X/Z detectors for proper L2 verification
        Q = num_ec_rounds
        samples = [x for x in samples if self.post_selector.post_selection_l2_memory(
            x, list_detector_0prep, list_detector_0prep_l2,
            list_detector_X, list_detector_Z, Q,
            detector_X_prep=detector_X_prep, detector_Z_prep=detector_Z_prep
        )]
        num = len(samples)
        
        # Count errors with hierarchical decoding
        # Get inner code size dynamically
        inner_n = self.concat_code.code_at_level(0).n
        
        num_errors = 0
        for x in samples:
            correction_x = np.zeros(inner_n)
            correction_z = np.zeros(inner_n)
            outer_x_correction = 0  # Accumulated outer code X correction
            
            # Decode EC rounds - decode_ec_hd returns (outer_x, outer_z, level1_x, level1_z)
            for i in range(num_ec_rounds):
                outer_x, outer_z, correction_x, correction_z = self.decoder.decode_ec_hd(
                    x, list_detector_X[i], list_detector_Z[i],
                    correction_x, correction_z
                )
                # Accumulate outer code X correction (X errors flip Z-basis measurements)
                if outer_x == 1:
                    outer_x_correction = (outer_x_correction + 1) % 2
            
            # Decode final Z-basis measurement with level-1 X corrections
            # X errors cause bit flips in Z-basis measurements!
            outcome = self.decoder.decode_m_hd(x, detector_m, correction_x)
            
            # Apply outer code X correction to the measurement outcome
            if outer_x_correction == 1:
                outcome = (outcome + 1) % 2
            if outcome != 0:
                num_errors += 1
        
        print(f"Memory L2: p={p}, accepted={num}, errors={num_errors}")
        
        if num > 0:
            logical_error = num_errors / num
            variance = num_errors / (num ** 2)
        else:
            logical_error = variance = 0
        
        return logical_error, variance


# =============================================================================
# Factory Functions
# ┌─────────────────────────────────────────────────────────────────────────────┐
# │                          FACTORY FUNCTIONS                                   │
# └─────────────────────────────────────────────────────────────────────────────┘

# Code Creation:
# ┌─────────────────────────────────────────────────────────────────────────────┐
# │ create_shor_code() -> CSSCode                                                │
# │   Returns Shor [[9,1,3]] with circuit specification                          │
# │                                                                              │
# │ create_concatenated_code(codes, prop_tables) -> ConcatenatedCode             │
# │   Generic factory for any concatenated code                                  │
# │                                                                              │
# │ For Steane-specific code, use concatenated_css_v10_steane.py:                │
# │   create_steane_code() -> CSSCode                                            │
# │   create_steane_propagation_l2() -> PropagationTables                        │
# │   create_concatenated_steane(num_levels) -> ConcatenatedCode                 │
# │   create_steane_simulator(num_levels, noise_model)                           │
# └─────────────────────────────────────────────────────────────────────────────┘

# Simulator Creation:
# ┌─────────────────────────────────────────────────────────────────────────────┐
# │ create_simulator(concat_code, noise_model)                                   │
# │   -> ConcatenatedCodeSimulator                                               │
# │   Uses generic preparation, EC, and decoding                                 │
# │                                                                              │
# │ For Steane-specific simulation, use:                                         │
# │   from concatenated_css_v10_steane import create_steane_simulator            │
# └─────────────────────────────────────────────────────────────────────────────┘
# =============================================================================

def create_simulator(concat_code: ConcatenatedCode, 
                     noise_model: NoiseModel) -> ConcatenatedCodeSimulator: 
    """
    Factory function to create a generic simulator.
    
    Args:
        concat_code: The concatenated code
        noise_model: Noise model to apply
    
    Returns:
        Configured simulator with generic components
    
    Note: For Steane-specific simulation, use create_steane_simulator from
    concatenated_css_v10_steane.py instead. That provides exact propagation tables
    for better decoding accuracy.
    
    This generic factory will auto-generate approximate propagation tables if
    none are provided, enabling L2+ simulation for any CSS code.
    """
    # Auto-generate propagation tables if not provided for L2+ codes
    if concat_code.num_levels >= 2 and 1 not in concat_code.propagation_tables:
        # Create a temporary prep strategy to compute tables
        ops = TransversalOps(concat_code)
        prep = GenericPreparationStrategy(concat_code, ops)
        inner_code = concat_code.code_at_level(0)
        
        # Compute approximate propagation tables
        prop_tables = prep.compute_propagation_tables(inner_code)
        concat_code.propagation_tables[1] = prop_tables
    
    return ConcatenatedCodeSimulator(concat_code, noise_model)


# =============================================================================
# Main Entry Point (for command-line usage)
# =============================================================================

if __name__ == '__main__':
    import sys
    import json
    
    print("For Steane code simulation, use concatenated_css_v10_steane.py")
    print("This module (concatenated_css_v10.py) contains only generic CSS code infrastructure.")
    print()
    print("Example usage:")
    print("  from concatenated_css_v10_steane import create_steane_simulator")
    print("  simulator = create_steane_simulator(num_levels=2, noise_model=noise_model)")
    print("  error, var = simulator.estimate_logical_cnot_error_l2(p=0.001, num_shots=10000)")