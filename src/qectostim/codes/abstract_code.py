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

Code Parameters:
    Derived classes define [[n, k, d]].  The abstract base only requires
    ``n`` (physical qubits) and ``k`` (logical qubits); ``d`` (distance)
    is optional metadata that concrete codes populate.

Stabiliser Structure:
    Derived classes provide Hx, Hz (for CSS codes) or a full symplectic
    stabiliser matrix (for non-CSS codes).  The base ``Code`` ABC imposes
    no stabiliser-specific constraints.

Raises:
    NotImplementedError
        Instantiating ``Code`` or ``StabilizerCode`` directly (they are
        abstract) or calling stabiliser accessors on a non-stabiliser code.
    TypeError
        If a subclass fails to implement the required abstract properties
        (``n``, ``k``, ``logical_x_ops``, ``logical_z_ops``).
"""
from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .abstract_css import CSSCode

PauliString = Dict[int, str]  # e.g. {0: 'X', 3: 'Z'} means X on qubit0, Z on qubit3
Coord = Tuple[float, ...]  # D-dimensional coordinates


def compute_coordinate_metadata(
    hx: np.ndarray,
    hz: np.ndarray,
    meta: Dict[str, Any],
    logical_x: Optional[List[PauliString]] = None,
    logical_z: Optional[List[PauliString]] = None,
) -> None:
    """Compute coordinate metadata from parity-check matrices and inject into *meta*.

    This utility is intended to be called by CSS-code constructors to populate
    ``data_coords``, ``x_stab_coords``, ``z_stab_coords``, ``lx_support``,
    and ``lz_support`` when they are not already present.

    For algebraically-constructed codes the data-qubit coordinates are laid
    out on a grid with ``cols = ceil(sqrt(n))`` columns.  Stabiliser coords
    are centroids of their support qubits.

    Parameters
    ----------
    hx : np.ndarray
        X-stabiliser parity-check matrix, shape ``(num_x_stabs, n)``.
    hz : np.ndarray
        Z-stabiliser parity-check matrix, shape ``(num_z_stabs, n)``.
    meta : dict
        Metadata dictionary to update **in-place**.
    logical_x, logical_z : list of PauliString, optional
        Logical operators; used to derive ``lx_support`` / ``lz_support``.
    """
    n = hx.shape[1]

    if "data_coords" not in meta:
        cols = int(np.ceil(np.sqrt(n)))
        data_coords_list = [(float(i % cols), float(i // cols)) for i in range(n)]
        meta["data_coords"] = data_coords_list
    else:
        data_coords_list = meta["data_coords"]

    if "x_stab_coords" not in meta:
        x_stab_coords_list = []
        for row_idx in range(hx.shape[0]):
            support = np.where(hx[row_idx])[0]
            if len(support) > 0:
                cx = np.mean([data_coords_list[q][0] for q in support])
                cy = np.mean([data_coords_list[q][1] for q in support])
                x_stab_coords_list.append((float(cx), float(cy)))
            else:
                x_stab_coords_list.append((0.0, 0.0))
        meta["x_stab_coords"] = x_stab_coords_list

    if "z_stab_coords" not in meta:
        z_stab_coords_list = []
        for row_idx in range(hz.shape[0]):
            support = np.where(hz[row_idx])[0]
            if len(support) > 0:
                cx = np.mean([data_coords_list[q][0] for q in support])
                cy = np.mean([data_coords_list[q][1] for q in support])
                z_stab_coords_list.append((float(cx), float(cy)))
            else:
                z_stab_coords_list.append((0.0, 0.0))
        meta["z_stab_coords"] = z_stab_coords_list

    if "lx_support" not in meta and logical_x:
        op = logical_x[0]
        if isinstance(op, dict):
            meta["lx_support"] = sorted(op.keys())
        elif isinstance(op, str):
            meta["lx_support"] = [i for i, c in enumerate(op) if c != 'I']

    if "lz_support" not in meta and logical_z:
        op = logical_z[0]
        if isinstance(op, dict):
            meta["lz_support"] = sorted(op.keys())
        elif isinstance(op, str):
            meta["lz_support"] = [i for i, c in enumerate(op) if c != 'I']


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
    def d(self) -> int:
        """
        Code distance as an integer (convenience alias).
        
        Returns the distance from metadata, defaulting to 3 if unknown.
        CSSCode overrides this with its own implementation.
        """
        dist = self.distance
        return dist if dist is not None else 3

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

    @property
    def is_stabilizer(self) -> bool:
        """Check if this code is a stabilizer code (has stabilizer_matrix)."""
        return False

    @property
    def is_self_dual(self) -> bool:
        """
        Whether the code is self-dual (X and Z stabilizers are identical).
        
        Only meaningful for CSS codes. Default is False.
        CSSCode overrides this with an actual hx==hz check.
        """
        return False

    def transversal_gates(self) -> List[str]:
        """
        List of gates this code supports transversally.
        
        Default uses metadata['transversal_gates'] if available,
        otherwise returns ["H", "CZ", "CNOT"] for CSS codes, [] otherwise.
        """
        meta_gates = self.metadata.get('transversal_gates', None)
        if meta_gates is not None:
            if isinstance(meta_gates, list):
                return meta_gates
            return []
        # Default for CSS codes
        if self.is_css:
            return ["H", "CZ", "CNOT"]
        return []

    def stabilizer_coords(self) -> Optional[List[Tuple[float, ...]]]:
        """
        Coordinates for stabilizer anchors (for detector coordinate assignment).
        
        Returns None by default. CSSCode overrides this to combine
        get_x_stabilizer_coords() and get_z_stabilizer_coords().
        """
        return None

    def extra_metadata(self) -> Dict[str, Any]:
        """Arbitrary metadata for advanced use."""
        return {}

    # --- Stabilizer accessors (standard interface) ---
    # These methods provide a CLEAN INTERFACE for accessing stabilizer information
    # without hasattr checks. Non-stabilizer codes should override to raise errors.
    
    def get_x_stabilizers(self) -> List[List[int]]:
        """
        Get X-type stabilizer supports.
        
        Returns a list of stabilizers, where each stabilizer is a list of
        qubit indices in its support. For CSS codes, these are the X-type
        stabilizers. For non-CSS codes, this returns stabilizers with X component.
        
        Returns
        -------
        List[List[int]]
            List of qubit index lists for each X-type stabilizer.
            
        Raises
        ------
        NotImplementedError
            If this code type doesn't support stabilizer access.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement get_x_stabilizers(). "
            "Override this method or use a StabilizerCode subclass."
        )
    
    def get_z_stabilizers(self) -> List[List[int]]:
        """
        Get Z-type stabilizer supports.
        
        Returns a list of stabilizers, where each stabilizer is a list of
        qubit indices in its support. For CSS codes, these are the Z-type
        stabilizers. For non-CSS codes, this returns stabilizers with Z component.
        
        Returns
        -------
        List[List[int]]
            List of qubit index lists for each Z-type stabilizer.
            
        Raises
        ------
        NotImplementedError
            If this code type doesn't support stabilizer access.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement get_z_stabilizers(). "
            "Override this method or use a StabilizerCode subclass."
        )
    
    def get_logical_x_support(self, logical_idx: int = 0) -> List[int]:
        """
        Get qubit support for a logical X operator.
        
        Parameters
        ----------
        logical_idx : int
            Which logical qubit (0 to k-1).
            
        Returns
        -------
        List[int]
            Qubit indices in the logical X support.
        """
        if logical_idx >= self.k:
            raise IndexError(f"Logical index {logical_idx} >= k={self.k}")
        x_op = self.logical_x_ops[logical_idx]
        return sorted(x_op.keys())
    
    def get_logical_z_support(self, logical_idx: int = 0) -> List[int]:
        """
        Get qubit support for a logical Z operator.
        
        Parameters
        ----------
        logical_idx : int
            Which logical qubit (0 to k-1).
            
        Returns
        -------
        List[int]
            Qubit indices in the logical Z support.
        """
        if logical_idx >= self.k:
            raise IndexError(f"Logical index {logical_idx} >= k={self.k}")
        z_op = self.logical_z_ops[logical_idx]
        return sorted(z_op.keys())

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
    def is_stabilizer(self) -> bool:
        """StabilizerCode always returns True."""
        return True
    
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
