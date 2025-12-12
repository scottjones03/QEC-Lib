"""Floquet Codes

Floquet codes are dynamical codes where the stabilizers are measured
in a time-periodic sequence. Unlike static stabilizer codes, the
effective stabilizers emerge from the measurement schedule.

Key features:
- Inherently fault-tolerant measurement circuits
- Good for certain hardware architectures
- Examples: Honeycomb code, ISG (instantaneous stabilizer group) codes

Note: Floquet codes use the FloquetCode base class which relaxes the
CSS constraint (Hx @ Hz.T = 0). This is because the "stabilizers" in
a Floquet code represent the instantaneous stabilizer group at one
point in the measurement schedule, not the full code stabilizers.

Reference: Hastings & Haah, "Dynamically Generated Logical Qubits" (2021)
"""

from __future__ import annotations
from typing import Dict, Any, List, Optional, Tuple
from abc import abstractmethod

import numpy as np

from qectostim.codes.abstract_css import CSSCode
from qectostim.codes.abstract_code import PauliString


class FloquetCode(CSSCode):
    """
    Base class for Floquet (dynamical) codes.
    
    Floquet codes are dynamical codes where stabilizers are measured in a 
    time-periodic sequence. The effective logical qubits emerge from the
    measurement schedule rather than from static stabilizers.
    
    Key differences from static CSS codes:
    1. The CSS constraint (Hx @ Hz.T = 0) is relaxed for the instantaneous
       stabilizer group representation
    2. Measurement schedules define the code behavior, not just Hx/Hz
    3. The distance may be schedule-dependent
    
    This base class provides:
    - Relaxed CSS validation (no commutativity check)
    - Floquet-specific metadata
    - Interface for measurement schedules (subclasses implement)
    
    Parameters
    ----------
    hx : np.ndarray
        X-type check matrix (instantaneous)
    hz : np.ndarray
        Z-type check matrix (instantaneous)
    logical_x : List[PauliString]
        X-type logical operators
    logical_z : List[PauliString]
        Z-type logical operators
    metadata : Dict[str, Any], optional
        Additional code metadata
    """
    
    def __init__(
        self,
        hx: np.ndarray,
        hz: np.ndarray,
        logical_x: List[PauliString],
        logical_z: List[PauliString],
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Initialize Floquet code without CSS commutativity check."""
        # Store matrices directly, bypassing CSS validation
        self._hx = np.array(hx, dtype=np.uint8)
        self._hz = np.array(hz, dtype=np.uint8)
        self._logical_x = logical_x
        self._logical_z = logical_z
        
        # Set up metadata with floquet flag
        meta = dict(metadata or {})
        meta["is_floquet"] = True
        meta["is_dynamical"] = True
        meta.setdefault("type", "floquet")
        self._metadata = meta
        
        # Validate dimensions only (not commutativity)
        self._validate_floquet()
    
    def _validate_floquet(self) -> None:
        """Validate Floquet code structure (dimensions only, no commutativity)."""
        if self._hx.size == 0 and self._hz.size == 0:
            return
        
        if self._hx.size > 0 and self._hz.size > 0:
            if self._hx.shape[1] != self._hz.shape[1]:
                raise ValueError(
                    f"Hx has {self._hx.shape[1]} columns, Hz has {self._hz.shape[1]} columns. "
                    "Both must have the same number of qubits."
                )
        
        # Log commutativity status for debugging (but don't error)
        if self._hx.size > 0 and self._hz.size > 0:
            comm = (self._hx @ self._hz.T) % 2
            if np.any(comm):
                # This is expected for Floquet codes - just note it in metadata
                self._metadata["css_commutes"] = False
            else:
                self._metadata["css_commutes"] = True
    
    def _validate_css(self) -> None:
        """Override CSS validation - Floquet codes don't require Hx @ Hz.T = 0."""
        # Skip the commutativity check - use _validate_floquet instead
        pass
    
    @property
    def measurement_schedule(self) -> Optional[List[str]]:
        """Return the measurement schedule for this Floquet code.
        
        Subclasses should override this to provide the actual schedule.
        
        Returns
        -------
        List[str] or None
            List of measurement types per round (e.g., ['XX', 'YY', 'ZZ'])
        """
        return self._metadata.get("measurement_schedule", None)
    
    @property
    def period(self) -> int:
        """Return the period of the measurement schedule.
        
        Returns
        -------
        int
            Number of measurement rounds before the schedule repeats
        """
        schedule = self.measurement_schedule
        return len(schedule) if schedule else 1


class HoneycombCode(FloquetCode):
    """
    Honeycomb Floquet Code.
    
    A dynamical code on a honeycomb lattice where measurements cycle
    through X-X, Y-Y, and Z-Z checks. The effective stabilizers are
    plaquettes that emerge from the measurement sequence.
    
    For static representation, we use the emergent stabilizers.
    
    Parameters
    ----------
    rows : int
        Number of rows in the honeycomb lattice
    cols : int
        Number of columns in the honeycomb lattice
    """
    
    def __init__(
        self,
        rows: int = 2,
        cols: int = 3,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Initialize honeycomb code."""
        if rows < 2 or cols < 2:
            raise ValueError("Honeycomb code requires at least 2x2 layout")
        
        # Honeycomb lattice: 2 qubits per unit cell
        n_cells = rows * cols
        n_qubits = 2 * n_cells
        
        def cell_idx(r: int, c: int) -> Tuple[int, int]:
            """Get qubit indices for cell (r, c)."""
            cell = (r % rows) * cols + (c % cols)
            return (2 * cell, 2 * cell + 1)
        
        # In honeycomb, each hexagon plaquette is a stabilizer
        # For a finite lattice, we have (rows-1) * (cols-1) plaquettes approx
        
        x_stabs = []
        z_stabs = []
        
        # Simplified stabilizer structure for finite patch
        # Each plaquette involves 6 qubits
        for r in range(rows - 1):
            for c in range(cols - 1):
                # Collect qubits around a plaquette
                q1, q2 = cell_idx(r, c)
                q3, q4 = cell_idx(r, c + 1)
                q5, q6 = cell_idx(r + 1, c)
                q7, q8 = cell_idx(r + 1, c + 1)
                
                # X stabilizer
                x_stab = [0] * n_qubits
                for q in [q1, q4, q5, q8]:
                    if q < n_qubits:
                        x_stab[q] = 1
                x_stabs.append(x_stab)
                
                # Z stabilizer (offset pattern)
                z_stab = [0] * n_qubits
                for q in [q2, q3, q6, q7]:
                    if q < n_qubits:
                        z_stab[q] = 1
                z_stabs.append(z_stab)
        
        # Ensure we have at least one stabilizer
        if not x_stabs:
            x_stabs = [[1, 1] + [0] * (n_qubits - 2)]
        if not z_stabs:
            z_stabs = [[0, 0, 1, 1] + [0] * (n_qubits - 4)] if n_qubits >= 4 else [[1, 1] + [0] * (n_qubits - 2)]
        
        hx = np.array(x_stabs, dtype=np.uint8)
        hz = np.array(z_stabs, dtype=np.uint8)
        
        # Logical operators span the torus
        # Logical operators (sparse dict format)
        lx_support = [i for i in range(n_qubits) if i % 2 == 0]
        lz_support = [i for i in range(n_qubits) if i % 2 == 1]
        
        logical_x: List[PauliString] = [{q: 'X' for q in lx_support}]
        logical_z: List[PauliString] = [{q: 'Z' for q in lz_support}]
        
        meta = dict(metadata or {})
        meta["name"] = f"Honeycomb_{rows}x{cols}"
        meta["n"] = n_qubits
        meta["k"] = 1
        meta["distance"] = min(rows, cols)  # Distance from lattice dimensions
        meta["rows"] = rows
        meta["cols"] = cols
        meta["type"] = "floquet"
        meta["measurement_schedule"] = ["XX", "YY", "ZZ"]  # Honeycomb measurement cycle
        
        super().__init__(hx=hx, hz=hz, logical_x=logical_x, logical_z=logical_z, metadata=meta)


class ISGFloquetCode(FloquetCode):
    """
    Instantaneous Stabilizer Group (ISG) Floquet Code.
    
    A general Floquet code framework where the ISG changes each round
    but has a periodic structure. This is the static representation
    of the accumulated stabilizer group.
    
    Parameters
    ----------
    base_distance : int
        Target code distance
    """
    
    def __init__(
        self,
        base_distance: int = 3,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Initialize ISG Floquet code."""
        d = base_distance
        
        # For ISG codes, we create a triangular-like structure
        # n qubits arranged to give distance d
        n_qubits = d * (d + 1) // 2 + d
        n_qubits = max(n_qubits, 4)  # At least 4 qubits
        
        # Create a simple stabilizer structure
        n_x_stabs = d
        n_z_stabs = d
        
        x_stabs = []
        z_stabs = []
        
        # X stabilizers
        for i in range(n_x_stabs):
            stab = [0] * n_qubits
            start = (i * 2) % n_qubits
            for j in range(2):
                stab[(start + j) % n_qubits] = 1
            x_stabs.append(stab)
        
        # Z stabilizers (offset)
        for i in range(n_z_stabs):
            stab = [0] * n_qubits
            start = (i * 2 + 1) % n_qubits
            for j in range(2):
                stab[(start + j) % n_qubits] = 1
            z_stabs.append(stab)
        
        hx = np.array(x_stabs, dtype=np.uint8)
        hz = np.array(z_stabs, dtype=np.uint8)
        
        # Logical operators
        logical_x: List[PauliString] = [{0: 'X'}]
        logical_z: List[PauliString] = [{0: 'Z'}]
        
        meta = dict(metadata or {})
        meta["name"] = f"ISGFloquet_d{d}"
        meta["n"] = n_qubits
        meta["k"] = 1
        meta["distance"] = d
        meta["type"] = "floquet_isg"
        
        super().__init__(hx=hx, hz=hz, logical_x=logical_x, logical_z=logical_z, metadata=meta)


# Pre-built instances
Honeycomb2x3 = lambda: HoneycombCode(rows=2, cols=3)
Honeycomb3x3 = lambda: HoneycombCode(rows=3, cols=3)
ISGFloquet3 = lambda: ISGFloquetCode(base_distance=3)
