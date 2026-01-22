# src/qectostim/experiments/hardware_simulation/core/sat_interface.py
"""
Abstract SAT solver interface for constraint-based optimization.

Provides a technology-agnostic interface for SAT-based routing and
placement optimization. Platform implementations can use different
SAT solvers (pysat, z3, etc.) while adhering to this interface.

Design Principles:
- No platform-specific terminology (no "ions", "atoms", "qubits" in core)
- Uses generic "items" and "positions" 
- Supports incremental solving and backtracking
- Provides hooks for soft constraints and optimization
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    Dict,
    List,
    Optional,
    Tuple,
    Any,
    Set,
    Callable,
    TypeVar,
    Generic,
    Protocol,
    runtime_checkable,
)


# =============================================================================
# Core SAT Routing Configuration
# =============================================================================

@dataclass
class SATRoutingConfig:
    """Technology-agnostic configuration for SAT-based routing.
    
    This base configuration contains parameters common to all platforms
    that use SAT solvers for routing optimization. Platform-specific
    configurations should extend this class.
    
    Attributes
    ----------
    timeout_seconds : float
        SAT solver timeout per configuration.
    max_passes : int
        Maximum routing passes per round.
    use_maxsat : bool
        If True, use MaxSAT for optimization; else pure SAT.
    debug_mode : bool
        Enable verbose logging.
    num_workers : int
        Number of parallel SAT workers.
    incremental : bool
        Use incremental SAT solving for faster subsequent solves.
    
    Example
    -------
    Platform-specific extensions add their own parameters:
    
    >>> @dataclass
    ... class WISERoutingConfig(SATRoutingConfig):
    ...     patch_enabled: bool = False
    ...     patch_height: int = 4
    ...     bt_soft_weight: int = 0  # WISE-specific
    """
    timeout_seconds: float = 60.0
    max_passes: int = 10
    use_maxsat: bool = True
    debug_mode: bool = False
    num_workers: int = 1
    incremental: bool = False
    
    # Common optimization weights
    movement_weight: float = 1.0     # Weight for movement/transport cost
    gate_priority_weight: float = 1.0  # Weight for gate execution priority
    
    def validate(self) -> List[str]:
        """Validate configuration, returning list of errors."""
        errors = []
        if self.timeout_seconds <= 0:
            errors.append("timeout_seconds must be positive")
        if self.max_passes <= 0:
            errors.append("max_passes must be positive")
        if self.num_workers < 1:
            errors.append("num_workers must be at least 1")
        return errors
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "SATRoutingConfig":
        """Create config from a dictionary."""
        return cls(**{
            k: v for k, v in config_dict.items()
            if k in cls.__dataclass_fields__
        })


@dataclass
class PatchRoutingConfig:
    """Configuration for patch-based routing on large grids.
    
    When grids are too large for direct SAT solving, patch routing
    divides the grid into smaller patches that are solved independently.
    This configuration controls patch decomposition.
    
    Technology-agnostic - works for any grid-based architecture.
    
    Attributes
    ----------
    enabled : bool
        Whether to use patch routing.
    patch_height : int
        Height of each patch in grid units.
    patch_width : int
        Width of each patch in grid units.
    overlap : int
        Overlap between adjacent patches for boundary handling.
    decomposition_strategy : str
        How to decompose into patches ("checkerboard", "strip", "recursive").
    """
    enabled: bool = False
    patch_height: int = 4
    patch_width: int = 4
    overlap: int = 0
    decomposition_strategy: str = "checkerboard"
    
    def num_patches(self, grid_rows: int, grid_cols: int) -> int:
        """Calculate number of patches for a grid."""
        if not self.enabled:
            return 1
        rows = (grid_rows + self.patch_height - 1) // self.patch_height
        cols = (grid_cols + self.patch_width - 1) // self.patch_width
        return rows * cols


# =============================================================================
# Core SAT Types
# =============================================================================

class ConstraintType(Enum):
    """Types of constraints in SAT encoding."""
    HARD = auto()       # Must be satisfied
    SOFT = auto()       # Prefer to satisfy (optimization)
    ASSUMPTION = auto() # Temporary assumption for incremental solving


@dataclass
class Constraint:
    """A constraint in the SAT problem.
    
    Attributes
    ----------
    constraint_type : ConstraintType
        Whether this is a hard or soft constraint.
    description : str
        Human-readable description for debugging.
    weight : float
        Weight for soft constraints (higher = more important).
    group : Optional[str]
        Optional grouping for constraint management.
    """
    constraint_type: ConstraintType = ConstraintType.HARD
    description: str = ""
    weight: float = 1.0
    group: Optional[str] = None


@dataclass
class SATSolution:
    """Result of a SAT solve operation.
    
    Attributes
    ----------
    satisfiable : bool
        Whether a solution was found.
    item_positions : Dict[int, Tuple[int, ...]]
        Mapping from item ID to position coordinates.
    cost : float
        Cost of this solution (for optimization).
    solve_time : float
        Time taken to solve in seconds.
    statistics : Dict[str, Any]
        Solver statistics (conflicts, decisions, etc.).
    """
    satisfiable: bool = False
    item_positions: Dict[int, Tuple[int, ...]] = field(default_factory=dict)
    cost: float = float('inf')
    solve_time: float = 0.0
    statistics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PlacementRequirement:
    """Requirement for item placement.
    
    Technology-agnostic: works for ion pairs, qubit pairs, atom pairs, etc.
    
    Attributes
    ----------
    items : Tuple[int, ...]
        Item IDs that need to interact.
    max_distance : int
        Maximum allowed distance between items.
    must_be_adjacent : bool
        Whether items must be at distance 1.
    """
    items: Tuple[int, ...]
    max_distance: int = 1
    must_be_adjacent: bool = True


# =============================================================================
# Abstract SAT Encoder Interface
# =============================================================================

class SATEncoder(ABC):
    """Abstract interface for SAT-based placement/routing encoders.
    
    This is the core abstraction that platform implementations extend.
    The encoder translates placement/routing problems into SAT constraints
    without any platform-specific knowledge.
    
    Example Usage
    -------------
    >>> encoder = MySATEncoder(grid_rows=4, grid_cols=4)
    >>> encoder.add_items(range(8))  # 8 items to place
    >>> encoder.add_placement_variables()
    >>> encoder.add_permutation_constraints()
    >>> encoder.add_adjacency_requirements([
    ...     PlacementRequirement((0, 1), must_be_adjacent=True),
    ...     PlacementRequirement((2, 3), must_be_adjacent=True),
    ... ])
    >>> solution = encoder.solve(timeout=10.0)
    >>> if solution.satisfiable:
    ...     print(solution.item_positions)
    """
    
    @abstractmethod
    def add_items(self, item_ids: List[int]) -> None:
        """Register items to be placed/routed.
        
        Parameters
        ----------
        item_ids : List[int]
            IDs of items to manage.
        """
        ...
    
    @abstractmethod
    def add_placement_variables(self) -> None:
        """Create SAT variables for item placements.
        
        Creates variables of the form "item i is at position p".
        The position representation depends on the encoder.
        """
        ...
    
    @abstractmethod
    def add_permutation_constraints(self) -> None:
        """Add constraints ensuring valid permutation.
        
        Ensures:
        - Each item is at exactly one position
        - Each position has at most one item (or exactly one, depending on mode)
        """
        ...
    
    @abstractmethod
    def add_adjacency_requirements(
        self,
        requirements: List[PlacementRequirement],
    ) -> None:
        """Add constraints for item adjacency.
        
        Parameters
        ----------
        requirements : List[PlacementRequirement]
            Pairs/groups of items that need to be close.
        """
        ...
    
    @abstractmethod
    def solve(
        self,
        timeout: Optional[float] = None,
        assumptions: Optional[List[int]] = None,
    ) -> SATSolution:
        """Solve the SAT problem.
        
        Parameters
        ----------
        timeout : Optional[float]
            Maximum solve time in seconds.
        assumptions : Optional[List[int]]
            Literal assumptions for incremental solving.
            
        Returns
        -------
        SATSolution
            The solution (or unsatisfiable result).
        """
        ...
    
    def reset(self) -> None:
        """Reset the encoder to initial state."""
        pass
    
    def add_blocking_clause(self, solution: SATSolution) -> None:
        """Add clause to block a previous solution.
        
        Useful for enumerating all solutions or finding alternatives.
        """
        pass


# =============================================================================
# Grid-Specific SAT Encoder
# =============================================================================

class GridSATEncoder(SATEncoder):
    """Abstract SAT encoder for grid-based layouts.
    
    Extends SATEncoder with grid-specific operations like
    row/column sorting, phase constraints, and swap networks.
    
    This is still technology-agnostic - it knows about grids
    but not about trapped ions, superconducting qubits, etc.
    """
    
    def __init__(
        self,
        rows: int,
        cols: int,
        layers: int = 1,
    ):
        """Initialize grid encoder.
        
        Parameters
        ----------
        rows : int
            Number of rows in the grid.
        cols : int
            Number of columns.
        layers : int
            Number of layers (for 3D grids).
        """
        self.rows = rows
        self.cols = cols
        self.layers = layers
        self._items: List[int] = []
    
    @property
    def num_sites(self) -> int:
        """Total number of grid sites."""
        return self.rows * self.cols * self.layers
    
    def add_items(self, item_ids: List[int]) -> None:
        """Register items to be placed on the grid."""
        self._items = list(item_ids)
    
    @abstractmethod
    def add_row_sorting_constraints(
        self,
        row: int,
        phase: int,
    ) -> None:
        """Add constraints for sorting items within a row.
        
        Used for odd-even transposition sort networks.
        
        Parameters
        ----------
        row : int
            Row index.
        phase : int
            Sort phase (affects which pairs compare).
        """
        ...
    
    @abstractmethod
    def add_column_sorting_constraints(
        self,
        col: int,
        phase: int,
    ) -> None:
        """Add constraints for sorting items within a column.
        
        Parameters
        ----------
        col : int
            Column index.
        phase : int
            Sort phase.
        """
        ...
    
    @abstractmethod
    def add_phase_constraints(
        self,
        phase_type: str,
        phase_number: int,
    ) -> None:
        """Add constraints for a routing phase.
        
        Phases typically alternate between horizontal (H) and
        vertical (V) operations in odd-even sorting.
        
        Parameters
        ----------
        phase_type : str
            Phase type identifier (e.g., "H", "V", "horizontal", "vertical").
        phase_number : int
            Phase number in sequence.
        """
        ...
    
    def get_position_coords(self, site_index: int) -> Tuple[int, int, int]:
        """Convert linear site index to (row, col, layer) coordinates."""
        layer = site_index // (self.rows * self.cols)
        remaining = site_index % (self.rows * self.cols)
        row = remaining // self.cols
        col = remaining % self.cols
        return (row, col, layer)
    
    def get_site_index(self, row: int, col: int, layer: int = 0) -> int:
        """Convert (row, col, layer) to linear site index."""
        return layer * self.rows * self.cols + row * self.cols + col


# =============================================================================
# Solver Protocol
# =============================================================================

class SATSolverProtocol(ABC):
    """Protocol for SAT solver backends.
    
    Allows injection of different SAT solvers (pysat, z3, etc.)
    into the encoding infrastructure.
    """
    
    @abstractmethod
    def add_clause(self, clause: List[int]) -> None:
        """Add a clause to the solver."""
        ...
    
    @abstractmethod
    def solve(
        self,
        assumptions: Optional[List[int]] = None,
    ) -> bool:
        """Solve with optional assumptions.
        
        Returns
        -------
        bool
            True if satisfiable.
        """
        ...
    
    @abstractmethod
    def get_model(self) -> Optional[List[int]]:
        """Get the satisfying assignment if SAT.
        
        Returns
        -------
        Optional[List[int]]
            List of literals in the model, or None if UNSAT.
        """
        ...
    
    def interrupt(self) -> None:
        """Interrupt the solver (for timeout)."""
        pass
    
    def delete(self) -> None:
        """Clean up solver resources."""
        pass


# =============================================================================
# Utility Functions
# =============================================================================

def manhattan_distance(
    pos1: Tuple[int, ...],
    pos2: Tuple[int, ...],
) -> int:
    """Compute Manhattan distance between two positions."""
    return sum(abs(a - b) for a, b in zip(pos1, pos2))


def are_adjacent(
    pos1: Tuple[int, ...],
    pos2: Tuple[int, ...],
) -> bool:
    """Check if two positions are adjacent (Manhattan distance 1)."""
    return manhattan_distance(pos1, pos2) == 1


def grid_neighbors(
    row: int,
    col: int,
    rows: int,
    cols: int,
    include_diagonal: bool = False,
) -> List[Tuple[int, int]]:
    """Get valid neighbor positions on a grid."""
    offsets = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    if include_diagonal:
        offsets += [(1, 1), (1, -1), (-1, 1), (-1, -1)]
    
    neighbors = []
    for dr, dc in offsets:
        nr, nc = row + dr, col + dc
        if 0 <= nr < rows and 0 <= nc < cols:
            neighbors.append((nr, nc))
    return neighbors
