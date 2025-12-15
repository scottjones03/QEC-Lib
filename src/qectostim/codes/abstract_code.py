# src/qectostim/codes/abstract_code.py
"""
Abstract base classes for quantum error-correcting codes.

Class Hierarchy:
    Code (ABC) - Base for any quantum code
    ├── StabilizerCode (ABC) - Codes defined by stabilizer group
    │   └── SubsystemCode (ABC) - Codes with gauge degrees of freedom
    └── (See abstract_homological.py for HomologicalCode branch)

Key Design Principles:
1. Code is the minimal interface for experiments (n, k, logical ops)
2. StabilizerCode adds stabilizer_matrix (symplectic form)
3. Stabilizers() is NOT abstract on Code - only StabilizerCode has it
4. HomologicalCode (in abstract_homological.py) adds chain complex structure
5. CSSCode (in abstract_css.py) inherits from both StabilizerCode and HomologicalCode
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .abstract_css import CSSCode

PauliString = Dict[int, str]  # e.g. {0: 'X', 3: 'Z'} means X on qubit0, Z on qubit3
Coord = Tuple[float, ...]  # D-dimensional coordinates


# =============================================================================
# FT Gadget Experiment Configuration
# =============================================================================

class ScheduleMode(Enum):
    """How to schedule CNOT gates in stabilizer measurement rounds."""
    AUTO = "auto"           # Try geometric first, fall back to graph-coloring
    GEOMETRIC = "geometric" # Coordinate-based scheduling (requires qubit coords)
    GRAPH_COLORING = "graph_coloring"  # Conflict-free graph coloring (always works)


@dataclass
class FTGadgetCodeConfig:
    """
    Configuration for how a code should behave in FT gadget experiments.
    
    This allows codes to customize their behavior without modifying the
    experiment or builder classes. Codes override get_ft_gadget_config()
    to return their specific configuration.
    
    Attributes
    ----------
    schedule_mode : ScheduleMode
        How to schedule CNOT gates in stabilizer rounds.
        - AUTO: Try geometric scheduling first, fall back to graph-coloring
        - GEOMETRIC: Require coordinate-based scheduling (fails if coords unavailable)
        - GRAPH_COLORING: Always use graph coloring (robust but less efficient)
        
    first_round_x_detectors : bool
        Whether to emit X stabilizer detectors on the first round.
        True for codes where X stabilizers are deterministic from start.
        False if initial state makes X stabilizers random.
        
    first_round_z_detectors : bool
        Whether to emit Z stabilizer detectors on the first round.
        True for codes where Z stabilizers are deterministic from start.
        False if initial state makes Z stabilizers random.
        
    enable_metachecks : bool
        Whether to use metachecks for single-shot error correction.
        Only applies to codes with chain_length >= 4 (3D+ codes).
        
    project_coords_to_2d : bool
        Whether to project higher-dimensional coordinates to 2D for visualization.
        Useful for 3D/4D codes where Stim expects 2D detector coords.
        
    supports_transversal_h : bool
        Whether the code supports transversal Hadamard naturally.
        Some codes (like XZZX surface) have modified stabilizer structures.
        
    extra : Dict[str, Any]
        Additional code-specific configuration.
    """
    schedule_mode: ScheduleMode = ScheduleMode.AUTO
    first_round_x_detectors: bool = True
    first_round_z_detectors: bool = True
    enable_metachecks: bool = False
    project_coords_to_2d: bool = False
    supports_transversal_h: bool = True
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CellEmbedding:
    """Embedding of cells of a given grade into D-dimensional space."""
    grade: int
    coords: List[Coord]   # length = dim(C_grade)


class Code(ABC):
    """
    Abstract base class for ANY quantum code.
    
    This is the minimal interface that experiments and decoders use.
    It does NOT require stabilizers() - that's only for StabilizerCode subclasses.
    
    Some codes (e.g., bosonic codes, qudit codes) may not have a stabilizer
    formalism but still have logical operators and encode information.
    """

    @property
    @abstractmethod
    def n(self) -> int:
        """Number of physical qubits."""

    @property
    @abstractmethod
    def k(self) -> int:
        """Number of logical qubits."""

    @property
    def distance(self) -> Optional[int]:
        """
        Code distance (minimum weight of logical operator).
        
        Returns None if unknown or not computed.
        """
        return self.metadata.get('distance', None)

    @property
    def name(self) -> str:
        return self.__class__.__name__

    @property
    def metadata(self) -> Dict[str, Any]:
        """Arbitrary metadata associated with the code instance."""
        return getattr(self, "_metadata", {})

    @metadata.setter
    def metadata(self, value: Dict[str, Any]) -> None:
        self._metadata = value

    # --- Logical operators ---

    @property
    @abstractmethod
    def logical_x_ops(self) -> List[PauliString]:
        """Logical X operators (one per logical qubit)."""

    @property
    @abstractmethod
    def logical_z_ops(self) -> List[PauliString]:
        """Logical Z operators (one per logical qubit)."""

    # --- Geometry (optional) ---

    def qubit_coords(self) -> Optional[List[Tuple[float, ...]]]:
        """
        Optional D-dimensional embedding for visualization.
        None if not defined.
        """
        return None

    # --- Type checking helpers ---

    def as_css(self) -> Optional["CSSCode"]:
        """Return self as a CSSCode if applicable, else None."""
        return None

    @property
    def is_css(self) -> bool:
        """Check if this code is a CSS code."""
        return False

    def extra_metadata(self) -> Dict[str, Any]:
        """Arbitrary metadata for advanced use."""
        return {}

    # --- FT Gadget Experiment Interface ---
    
    def get_ft_gadget_config(self) -> FTGadgetCodeConfig:
        """
        Get configuration for FT gadget experiments.
        
        Override this method in subclasses to customize behavior.
        The default configuration works for most standard CSS codes.
        
        Returns
        -------
        FTGadgetCodeConfig
            Configuration for stabilizer rounds, scheduling, etc.
            
        Examples
        --------
        >>> # Default for most codes
        >>> code.get_ft_gadget_config()
        FTGadgetCodeConfig(schedule_mode=AUTO, ...)
        
        >>> # Override in subclass for special behavior
        >>> class XZZXSurfaceCode(CSSCode):
        ...     def get_ft_gadget_config(self):
        ...         return FTGadgetCodeConfig(
        ...             schedule_mode=ScheduleMode.GRAPH_COLORING,
        ...         )
        """
        return FTGadgetCodeConfig()
    
    def validate_for_ft_experiment(self) -> Tuple[bool, str]:
        """
        Validate that this code can be used in FT gadget experiments.
        
        Override in subclasses to add code-specific validation.
        
        Returns
        -------
        Tuple[bool, str]
            (is_valid, error_message) - is_valid is True if code is usable.
        """
        # Basic validation
        if self.n <= 0:
            return False, "Code must have at least 1 physical qubit"
        if self.k <= 0:
            return False, "Code must encode at least 1 logical qubit"
        return True, ""
    
    def project_coords_to_2d(
        self,
        coords: List[Tuple[float, ...]],
    ) -> List[Tuple[float, float]]:
        """
        Project higher-dimensional coordinates to 2D for visualization.
        
        Default implementation takes first two coordinates.
        Override for codes that need custom projection (e.g., 3D toric).
        
        Parameters
        ----------
        coords : List[Tuple[float, ...]]
            Original D-dimensional coordinates.
            
        Returns
        -------
        List[Tuple[float, float]]
            Projected 2D coordinates.
        """
        result = []
        for c in coords:
            if len(c) >= 2:
                result.append((float(c[0]), float(c[1])))
            elif len(c) == 1:
                result.append((float(c[0]), 0.0))
            else:
                result.append((0.0, 0.0))
        return result


class StabilizerCode(Code):
    """
    Abstract base class for stabilizer codes (both CSS and non-CSS).
    
    A stabilizer code is defined by:
    - A stabilizer group S ⊂ Pauli_n generated by commuting Pauli operators
    - Logical operators in normalizer(S) \ S
    
    The stabilizer generators are stored in symplectic form as a matrix
    [X_part | Z_part] where each row is a stabilizer generator.
    
    For a code on n qubits with m stabilizer generators:
    - stabilizer_matrix has shape (m, 2*n)
    - First n columns: X components (1 if X or Y on that qubit)
    - Last n columns: Z components (1 if Z or Y on that qubit)
    
    Subclasses:
    - CSSCode: Stabilizers factor into pure X-type and pure Z-type
    - SubsystemCode: Has gauge operators in addition to stabilizers
    """
    
    @property
    @abstractmethod
    def stabilizer_matrix(self) -> np.ndarray:
        """
        Stabilizer generators in symplectic form.
        
        Returns
        -------
        np.ndarray
            Shape (m, 2*n) where m = number of stabilizers, n = number of qubits.
            Format: [X_part | Z_part] where X_part[i,j]=1 means X on qubit j,
            Z_part[i,j]=1 means Z on qubit j. Both 1 means Y.
        """
    
    @property
    def is_css(self) -> bool:
        """
        Check if this is a CSS code (all stabilizers are pure X-type or pure Z-type).
        
        Returns True if every stabilizer generator has support on only X or only Z
        (but not both, except for identity).
        """
        stab_mat = self.stabilizer_matrix
        if stab_mat.size == 0:
            return True
        
        n = self.n
        x_part = stab_mat[:, :n]
        z_part = stab_mat[:, n:]
        
        # Check each stabilizer: it's CSS if it has ONLY X support OR ONLY Z support
        for i in range(stab_mat.shape[0]):
            has_x = np.any(x_part[i] != 0)
            has_z = np.any(z_part[i] != 0)
            if has_x and has_z:
                return False
        return True
    
    def stabilizers(self) -> List[PauliString]:
        """
        Convert stabilizer matrix to list of Pauli strings.
        
        Returns
        -------
        List[PauliString]
            Each dict maps qubit index to Pauli type ('X', 'Y', 'Z').
        """
        stab_mat = self.stabilizer_matrix
        if stab_mat.size == 0:
            return []
        
        n = self.n
        x_part = stab_mat[:, :n]
        z_part = stab_mat[:, n:]
        
        stabs = []
        for i in range(stab_mat.shape[0]):
            pauli: PauliString = {}
            for j in range(n):
                x_bit = x_part[i, j]
                z_bit = z_part[i, j]
                if x_bit and z_bit:
                    pauli[j] = 'Y'
                elif x_bit:
                    pauli[j] = 'X'
                elif z_bit:
                    pauli[j] = 'Z'
            stabs.append(pauli)
        return stabs
    
    def x_stabilizers(self) -> List[PauliString]:
        """Get stabilizers that are pure X-type (only X operators)."""
        return [s for s in self.stabilizers() if s and all(p == 'X' for p in s.values())]
    
    def z_stabilizers(self) -> List[PauliString]:
        """Get stabilizers that are pure Z-type (only Z operators)."""
        return [s for s in self.stabilizers() if s and all(p == 'Z' for p in s.values())]
    
    def mixed_stabilizers(self) -> List[PauliString]:
        """Get stabilizers that have both X and Z components (non-CSS type)."""
        stabs = self.stabilizers()
        mixed = []
        for s in stabs:
            if not s:
                continue
            paulis = set(s.values())
            if len(paulis) > 1 or 'Y' in paulis:
                mixed.append(s)
        return mixed

    def gauge_ops(self) -> List[PauliString]:
        """Gauge operators (empty for non-subsystem codes)."""
        return []


class SubsystemCode(StabilizerCode):
    """
    Abstract base class for subsystem codes.
    
    A subsystem code has:
    - Gauge group G ⊂ Pauli_n: contains gauge operators (not all commute)
    - Stabilizer group S = center(G) ∩ Pauli_n: center of gauge group
    - Logical operators: operators in normalizer(S) but not in G
    
    The Hilbert space decomposes as: H = H_L ⊗ H_G ⊗ H_S
    where H_L is the logical space, H_G is gauge space, H_S is syndrome space.
    
    The key advantage is that syndrome extraction only requires measuring
    gauge operators (often weight-2) rather than full stabilizers.
    """
    
    @property
    @abstractmethod
    def gauge_matrix(self) -> np.ndarray:
        """
        Gauge operators in symplectic form.
        
        Returns
        -------
        np.ndarray
            Shape (g, 2*n) where g = number of gauge generators.
            Same format as stabilizer_matrix.
        """
    
    def gauge_ops(self) -> List[PauliString]:
        """Convert gauge matrix to list of Pauli strings."""
        gauge_mat = self.gauge_matrix
        if gauge_mat.size == 0:
            return []
        
        n = self.n
        x_part = gauge_mat[:, :n]
        z_part = gauge_mat[:, n:]
        
        gauges = []
        for i in range(gauge_mat.shape[0]):
            pauli: PauliString = {}
            for j in range(n):
                x_bit = x_part[i, j]
                z_bit = z_part[i, j]
                if x_bit and z_bit:
                    pauli[j] = 'Y'
                elif x_bit:
                    pauli[j] = 'X'
                elif z_bit:
                    pauli[j] = 'Z'
            gauges.append(pauli)
        return gauges
