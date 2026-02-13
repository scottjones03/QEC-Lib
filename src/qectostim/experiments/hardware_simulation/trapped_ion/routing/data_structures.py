# src/qectostim/experiments/hardware_simulation/trapped_ion/routing/data_structures.py
"""
WISE routing data structures.

This module provides ion-specific wrappers around core routing structures:
- GridLayout: Ion layout on a 2D WISE grid
- RoutingPass: A single H or V routing pass
- RoutingSchedule: Complete sequence of routing passes
- GatePairRequirement: A two-qubit gate requirement
- compute_target_positions: Compute target positions for gate pairs
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Set, Tuple

import numpy as np

from qectostim.experiments.hardware_simulation.core.sat_interface import (
    GridLayout as CoreGridLayout,
    SortingPass as CoreSortingPass,
    RoutingSchedule as CoreRoutingSchedule,
    InteractionRequirement as CoreInteractionRequirement,
)


# =============================================================================
# GridLayout - Ion-specific wrapper
# =============================================================================

class GridLayout(CoreGridLayout):
    """Ion layout on a 2D WISE grid.
    
    Inherits from core.sat_interface.GridLayout with ion-specific aliases.
    
    Attributes
    ----------
    grid : np.ndarray
        2D array of ion indices (n_rows x n_cols).
    ion_positions : Dict[int, Tuple[int, int]]
        Mapping from ion index to (row, col) position.
    """
    
    @property
    def ion_positions(self) -> Dict[int, Tuple[int, int]]:
        """Alias for item_positions using ion terminology."""
        return self.item_positions
    
    def get_ion_at(self, row: int, col: int) -> int:
        """Get ion index at position (alias for get_item_at)."""
        return self.get_item_at(row, col)
    
    def copy(self) -> "GridLayout":
        """Create a deep copy."""
        return GridLayout(
            grid=self.grid.copy(),
            item_positions=dict(self.item_positions),
        )


# =============================================================================
# RoutingPass - Alias for SortingPass
# =============================================================================

class RoutingPass(CoreSortingPass):
    """A single routing pass (H or V phase) for WISE routing.
    
    Inherits from core.sat_interface.SortingPass.
    """
    pass


# =============================================================================
# RoutingSchedule - Direct re-export
# =============================================================================

RoutingSchedule = CoreRoutingSchedule


# =============================================================================
# GatePairRequirement
# =============================================================================

@dataclass
class GatePairRequirement(CoreInteractionRequirement):
    """Requirement for a two-qubit gate in WISE routing.
    
    Extends core.sat_interface.InteractionRequirement with gate-specific naming.
    
    Attributes
    ----------
    ion_a : int
        First ion index.
    ion_b : int
        Second ion index.
    round_idx : int
        Which round this gate is in.
    gate_type : str
        Type of gate (e.g., "MS", "CZ").
    """
    gate_type: str = "MS"
    
    @property
    def ion_a(self) -> int:
        """Alias for item_a using ion terminology."""
        return self.item_a
    
    @property
    def ion_b(self) -> int:
        """Alias for item_b using ion terminology."""
        return self.item_b


# =============================================================================
# Gate Pair Analysis
# =============================================================================

def compute_target_positions(
    pairs: List[Tuple[int, int]],
    n_rows: int,
    n_cols: int,
    capacity: int = 2,
) -> Dict[int, Tuple[int, int]]:
    """Compute target positions for ions to satisfy gate pairs.
    
    Uses bipartite matching to assign ions to gating positions.
    
    Parameters
    ----------
    pairs : List[Tuple[int, int]]
        Ion pairs that need to interact.
    n_rows : int
        Grid rows.
    n_cols : int
        Grid columns.
    capacity : int
        Block width for gating zones.
        
    Returns
    -------
    Dict[int, Tuple[int, int]]
        Target position for each ion involved in gates.
    """
    targets: Dict[int, Tuple[int, int]] = {}
    
    # Collect all ions involved
    all_ions: Set[int] = set()
    for a, b in pairs:
        all_ions.add(a)
        all_ions.add(b)
    
    # Available gating positions (pairs of adjacent cells)
    gating_positions: List[Tuple[Tuple[int, int], Tuple[int, int]]] = []
    for r in range(n_rows):
        for c in range(0, n_cols - 1, capacity):
            gating_positions.append(((r, c), (r, c + 1)))
    
    # Simple greedy assignment
    used_positions: Set[Tuple[int, int]] = set()
    
    for ion_a, ion_b in pairs:
        if ion_a in targets and ion_b in targets:
            continue
        
        # Find an unused gating position
        for pos_a, pos_b in gating_positions:
            if pos_a not in used_positions and pos_b not in used_positions:
                if ion_a not in targets:
                    targets[ion_a] = pos_a
                    used_positions.add(pos_a)
                if ion_b not in targets:
                    targets[ion_b] = pos_b
                    used_positions.add(pos_b)
                break
    
    return targets


__all__ = [
    "GridLayout",
    "RoutingPass",
    "RoutingSchedule",
    "GatePairRequirement",
    "compute_target_positions",
]
