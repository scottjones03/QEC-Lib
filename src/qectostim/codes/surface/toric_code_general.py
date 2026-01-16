"""General Toric Code [[2L², 2, L]]

The toric code is a topological CSS code defined on a torus (periodic boundary conditions).
For an LxL torus:
- 2L² qubits on the edges (L² horizontal + L² vertical)
- L² X-type (plaquette/face) stabilizers (L²-1 independent)
- L² Z-type (vertex/star) stabilizers (L²-1 independent)
- 2 logical qubits corresponding to the two non-contractible cycles

This generalizes ToricCode33 to arbitrary lattice sizes.
"""

from __future__ import annotations
from typing import Dict, Any, List, Optional, Tuple

import numpy as np

from qectostim.codes.abstract_css import TopologicalCSSCode, Coord2D
from qectostim.codes.abstract_code import PauliString, FTGadgetCodeConfig, ScheduleMode
from qectostim.codes.complexes.css_complex import CSSChainComplex3


class ToricCode(TopologicalCSSCode):
    """
    [[2L², 2, L]] Toric code on an LxL torus.

    2L² qubits (edges), L²-1 X-type plaquette checks, L²-1 Z-type vertex checks.
    Encodes 2 logical qubits with distance L.
    
    Inherits from TopologicalCSSCode with full chain complex structure.
    
    Parameters
    ----------
    Lx : int
        Lattice size in x direction. Must be >= 2.
    Ly : int, optional
        Lattice size in y direction. If None, uses Lx (square lattice).
    """

    def __init__(self, Lx: int, Ly: Optional[int] = None, metadata: Optional[Dict[str, Any]] = None):
        """Initialize the toric code with given lattice dimensions."""
        if Ly is None:
            Ly = Lx
            
        if Lx < 2 or Ly < 2:
            raise ValueError(f"Lattice dimensions must be >= 2, got Lx={Lx}, Ly={Ly}")
        
        n_qubits = 2 * Lx * Ly  # Lx*Ly horizontal + Lx*Ly vertical edges
        
        # Edge indexing:
        # Horizontal edge at (row, col): index = row * Ly + col
        # Vertical edge at (row, col): index = Lx*Ly + row * Ly + col
        def h_edge(row, col):
            return (row % Lx) * Ly + (col % Ly)
        
        def v_edge(row, col):
            return Lx * Ly + (row % Lx) * Ly + (col % Ly)
        
        # Z-type stabilizers (vertex/star operators)
        # Build as boundary_2_z: each vertex touches 4 edges
        hz_full = np.zeros((Lx * Ly, n_qubits), dtype=np.uint8)
        z_stab_coords = []
        for i in range(Lx):
            for j in range(Ly):
                vertex_idx = i * Ly + j
                hz_full[vertex_idx, h_edge(i, j)] = 1        # right horizontal
                hz_full[vertex_idx, h_edge(i, (j - 1) % Ly)] = 1  # left horizontal
                hz_full[vertex_idx, v_edge(i, j)] = 1        # down vertical
                hz_full[vertex_idx, v_edge((i - 1) % Lx, j)] = 1  # up vertical
                z_stab_coords.append((float(j), float(i)))  # vertex at integer coords
        
        hz = hz_full[:-1]  # Remove last dependent row
        
        # X-type stabilizers (plaquette/face operators)
        # Build as boundary_2_x: each plaquette has 4 edges
        hx_full = np.zeros((Lx * Ly, n_qubits), dtype=np.uint8)
        x_stab_coords = []
        for i in range(Lx):
            for j in range(Ly):
                plaq_idx = i * Ly + j
                hx_full[plaq_idx, h_edge(i, j)] = 1          # top horizontal
                hx_full[plaq_idx, h_edge((i + 1) % Lx, j)] = 1  # bottom horizontal
                hx_full[plaq_idx, v_edge(i, j)] = 1          # left vertical
                hx_full[plaq_idx, v_edge(i, (j + 1) % Ly)] = 1  # right vertical
                x_stab_coords.append((float(j) + 0.5, float(i) + 0.5))  # plaquette at half-integer
        
        hx = hx_full[:-1]  # Remove last dependent row
        
        # Build chain complex: C2 (faces) --∂2--> C1 (edges) --∂1--> C0 (vertices)
        # 
        # For toric code:
        # - ∂2: maps faces to their boundary edges (each face has 4 edges)
        #   Shape: (n_edges, n_faces) where n_faces = Lx*Ly
        # - ∂1: maps edges to their boundary vertices (each edge connects 2 vertices)
        #   Shape: (n_vertices, n_edges) where n_vertices = Lx*Ly
        #
        # CSS correspondence:
        # - Hx = ∂2^T (X stabilizers from face boundaries)
        # - Hz = ∂1 (Z stabilizers from edge-vertex incidence)
        #
        # Chain condition: ∂1 ∘ ∂2 = 0 (boundary of a boundary is zero)
        # This is satisfied because every face is a cycle (its boundary vertices cancel mod 2)
        
        # boundary_2: face → edge incidence (transpose of Hx_full)
        # Use full matrices to preserve chain condition
        boundary_2 = hx_full.T.astype(np.uint8)  # shape: (n_edges, n_faces)
        
        # boundary_1: edge → vertex incidence (Hz_full)
        # Each edge connects exactly 2 vertices
        boundary_1 = hz_full.astype(np.uint8)  # shape: (n_vertices, n_edges)
        
        chain_complex = CSSChainComplex3(boundary_2=boundary_2, boundary_1=boundary_1)
        
        # Logical operators (2 logical qubits)
        # Torus has two non-contractible cycles: around horizontal and vertical directions
        # Valid logical Z ops must be cycles that commute with all X stabilizers (plaquettes)
        # Analysis shows:
        #   - Vertical edges in a horizontal cycle (fixed row, all cols) commute with X stabs
        #   - Horizontal edges in a vertical cycle (all rows, fixed col) commute with X stabs
        
        # Logical X1: horizontal edges in row 0 (anti-commutes with L_Z1)
        lx1 = ['I'] * n_qubits
        for j in range(Ly):
            lx1[h_edge(0, j)] = 'X'
        
        # Logical Z1: vertical edges in row 0 (horizontal cycle around torus)
        # This is a cycle of vertical edges going "around" the horizontal direction
        lz1 = ['I'] * n_qubits
        for j in range(Ly):
            lz1[v_edge(0, j)] = 'Z'
        
        # Logical X2: vertical edges in row 0 (anti-commutes with L_Z2)
        lx2 = ['I'] * n_qubits
        for j in range(Ly):
            lx2[v_edge(0, j)] = 'X'
        
        # Logical Z2: horizontal edges in column 0 (vertical cycle around torus)
        # This is a cycle of horizontal edges going "around" the vertical direction
        lz2 = ['I'] * n_qubits
        for i in range(Lx):
            lz2[h_edge(i, 0)] = 'Z'
        
        logical_x = [''.join(lx1), ''.join(lx2)]
        logical_z = [''.join(lz1), ''.join(lz2)]

        # Distance is minimum of Lx, Ly
        distance = min(Lx, Ly)

        # Build qubit coordinates
        coords = {}
        for i in range(Lx):
            for j in range(Ly):
                coords[h_edge(i, j)] = (j + 0.5, float(i))
                coords[v_edge(i, j)] = (float(j), i + 0.5)
        data_coords = [coords.get(i, (0.0, 0.0)) for i in range(n_qubits)]
        
        meta = dict(metadata or {})
        meta["name"] = f"ToricCode_{Lx}x{Ly}"
        meta["n"] = n_qubits
        meta["k"] = 2
        meta["distance"] = distance
        meta["Lx"] = Lx
        meta["Ly"] = Ly
        meta["data_coords"] = data_coords
        meta["x_stab_coords"] = x_stab_coords[:-1]  # Match independent stabilizers
        meta["z_stab_coords"] = z_stab_coords[:-1]
        # Lattice size for periodic boundary wrapping in stabilizer rounds
        # For square lattices, lattice_size is sufficient
        # For non-square, we provide both dimensions
        meta["lattice_size"] = Lx if Lx == Ly else None
        meta["lattice_size_x"] = Lx
        meta["lattice_size_y"] = Ly
        
        # Measurement schedule for toric code (4-phase for weight-4 stabilizers)
        meta["x_schedule"] = [
            (0.0, 0.5),    # top edge
            (0.5, 0.0),    # left edge
            (0.0, -0.5),   # bottom edge
            (-0.5, 0.0),   # right edge
        ]
        meta["z_schedule"] = [
            (0.5, 0.0),    # right edge
            (0.0, 0.5),    # down edge
            (-0.5, 0.0),   # left edge
            (0.0, -0.5),   # up edge
        ]

        super().__init__(chain_complex, logical_x, logical_z, metadata=meta)
        
        # Override the parity check matrices for proper CSS structure
        self._hx = hx.astype(np.uint8)
        self._hz = hz.astype(np.uint8)
    
    def qubit_coords(self) -> List[Coord2D]:
        """Return qubit coordinates for visualization."""
        return list(self.metadata.get("data_coords", []))

    def get_ft_gadget_config(self) -> FTGadgetCodeConfig:
        """
        Return FT gadget configuration for toric codes.
        
        Toric codes have periodic boundary conditions which require special
        handling for coordinate lookups. We use GRAPH_COLORING scheduling
        to ensure robustness for all lattice sizes and gadget types.
        """
        return FTGadgetCodeConfig(
            schedule_mode=ScheduleMode.GRAPH_COLORING,  # Robust for periodic BCs
            first_round_x_detectors=True,
            first_round_z_detectors=True,
            enable_metachecks=False,
        )
