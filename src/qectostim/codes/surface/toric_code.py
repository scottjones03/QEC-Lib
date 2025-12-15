"""[[18,2,3]] Toric Code on 3×3 Torus

The toric code is a topological CSS code defined on a torus (periodic boundary conditions).
For a 3x3 torus:
- 18 qubits on the edges (9 horizontal + 9 vertical)
- 9 X-type (plaquette/face) stabilizers
- 9 Z-type (vertex/star) stabilizers
- 2 logical qubits corresponding to the two non-contractible cycles

The stabilizers are:
- X plaquettes: 4-body operators on edges around each face
- Z stars: 4-body operators on edges meeting at each vertex

Chain complex structure:
- C2 (faces/plaquettes) --∂2--> C1 (edges/qubits) --∂1--> C0 (vertices)
- X stabilizers come from ∂2 (plaquettes acting on boundary edges)
- Z stabilizers come from ∂1^T (vertices acting on incident edges)
"""

from __future__ import annotations
from typing import Dict, Any, List, Optional, Tuple

import numpy as np

from qectostim.codes.abstract_css import TopologicalCSSCode, Coord2D
from qectostim.codes.abstract_code import PauliString, FTGadgetCodeConfig, ScheduleMode
from qectostim.codes.complexes.css_complex import CSSChainComplex3


class ToricCode33(TopologicalCSSCode):
    """
    [[18,2,3]] Toric code on a 3×3 torus.

    18 qubits (edges), 8 X-type plaquette checks, 8 Z-type vertex checks.
    Encodes 2 logical qubits with distance 3.
    
    Inherits from TopologicalCSSCode with chain complex structure:
    C2 (faces) --∂2--> C1 (edges/qubits) --∂1--> C0 (vertices)
    """

    def __init__(self, metadata: Optional[Dict[str, Any]] = None):
        """Initialize the 3x3 toric code with chain complex structure.
        
        Qubit layout on 3x3 torus:
        - Horizontal edges: qubits 0-8 (h[row,col] at edge between vertices)
        - Vertical edges: qubits 9-17 (v[row,col] at edge between vertices)
        """
        L = 3  # Lattice size
        n_qubits = 2 * L * L  # 18 qubits
        
        # Build lattice geometry and parity check matrices
        (
            data_coords,
            x_stab_coords,
            z_stab_coords,
            hx,
            hz,
            boundary_2,
            boundary_1,
        ) = self._build_toric_lattice(L)
        
        # Create chain complex
        chain_complex = CSSChainComplex3(boundary_2=boundary_2, boundary_1=boundary_1)
        
        # Build logical operators
        logical_x, logical_z = self._build_logicals(L, n_qubits)
        
        # Metadata
        meta: Dict[str, Any] = dict(metadata or {})
        meta.update({
            "name": "ToricCode_3x3",
            "n": n_qubits,
            "k": 2,
            "distance": L,
            "lattice_size": L,
            "data_coords": data_coords,
            "x_stab_coords": x_stab_coords,
            "z_stab_coords": z_stab_coords,
            "logical_x_support": list(range(L)),
            "logical_z_support": list(range(L * L, L * L + L)),
        })
        
        # Measurement schedules for toric code (4-phase schedule)
        meta["x_schedule"] = [
            (0.5, 0.0),    # right horizontal edge
            (-0.5, 0.0),   # left horizontal edge
            (0.0, 0.5),    # bottom vertical edge
            (0.0, -0.5),   # top vertical edge
        ]
        meta["z_schedule"] = [
            (0.5, 0.0),    # right horizontal edge
            (0.0, 0.5),    # down vertical edge
            (-0.5, 0.0),   # left horizontal edge
            (0.0, -0.5),   # up vertical edge
        ]
        
        # Call TopologicalCSSCode constructor
        super().__init__(chain_complex, logical_x, logical_z, metadata=meta)
        
        # Override with explicit X/Z parity check matrices
        self._hx = hx
        self._hz = hz

    @staticmethod
    def _build_toric_lattice(L: int) -> Tuple[
        List[Coord2D],   # data_coords
        List[Coord2D],   # x_stab_coords
        List[Coord2D],   # z_stab_coords
        np.ndarray,      # hx
        np.ndarray,      # hz
        np.ndarray,      # boundary_2
        np.ndarray,      # boundary_1
    ]:
        """Build the toric code lattice, parity check matrices, and chain complex."""
        n_qubits = 2 * L * L
        
        # Edge indexing functions
        def h_edge(row, col):
            return (row % L) * L + (col % L)
        
        def v_edge(row, col):
            return L * L + (row % L) * L + (col % L)
        
        # Data qubit coordinates
        coords = {}
        for i in range(L):
            for j in range(L):
                coords[h_edge(i, j)] = (j + 0.5, float(i))
                coords[v_edge(i, j)] = (float(j), i + 0.5)
        data_coords = [coords.get(i, (0.0, 0.0)) for i in range(n_qubits)]
        
        # X-type stabilizers (plaquette/face operators)
        hx_full = np.zeros((L * L, n_qubits), dtype=np.uint8)
        for i in range(L):
            for j in range(L):
                plaq_idx = i * L + j
                hx_full[plaq_idx, h_edge(i, j)] = 1
                hx_full[plaq_idx, h_edge((i + 1) % L, j)] = 1
                hx_full[plaq_idx, v_edge(i, j)] = 1
                hx_full[plaq_idx, v_edge(i, (j + 1) % L)] = 1
        hx = hx_full[:-1]  # Remove last dependent row
        
        # Z-type stabilizers (vertex/star operators)
        hz_full = np.zeros((L * L, n_qubits), dtype=np.uint8)
        for i in range(L):
            for j in range(L):
                vertex_idx = i * L + j
                hz_full[vertex_idx, h_edge(i, j)] = 1
                hz_full[vertex_idx, h_edge(i, (j - 1) % L)] = 1
                hz_full[vertex_idx, v_edge(i, j)] = 1
                hz_full[vertex_idx, v_edge((i - 1) % L, j)] = 1
        hz = hz_full[:-1]  # Remove last dependent row
        
        # Build chain complex boundary matrices
        # ∂2: shape (n_edges, n_faces) - faces to edges incidence
        # For CSS: H_X = ∂2^T, so ∂2 = H_X^T
        boundary_2 = hx.T
        
        # ∂1: shape (n_vertices, n_edges) - edges to vertices incidence
        # For CSS: H_Z = ∂1, so ∂1 = H_Z
        boundary_1 = hz
        
        # Stabilizer coordinates
        x_stab_coords = [(j + 0.5, i + 0.5) for i in range(L) for j in range(L)][:-1]
        z_stab_coords = [(float(j), float(i)) for i in range(L) for j in range(L)][:-1]
        
        return (data_coords, x_stab_coords, z_stab_coords, hx, hz, boundary_2, boundary_1)

    @staticmethod
    def _build_logicals(L: int, n_qubits: int) -> Tuple[List[str], List[str]]:
        """Build logical operators for toric code."""
        def h_edge(row, col):
            return (row % L) * L + (col % L)
        
        def v_edge(row, col):
            return L * L + (row % L) * L + (col % L)
        
        # Logical X1: horizontal string (all horizontal edges in row 0)
        lx1 = ['I'] * n_qubits
        for j in range(L):
            lx1[h_edge(0, j)] = 'X'
        
        # Logical Z1: vertical string (all vertical edges in column 0)
        lz1 = ['I'] * n_qubits
        for i in range(L):
            lz1[v_edge(i, 0)] = 'Z'
        
        # Logical X2: vertical string (all vertical edges in row 0)
        lx2 = ['I'] * n_qubits
        for j in range(L):
            lx2[v_edge(0, j)] = 'X'
        
        # Logical Z2: horizontal string (all horizontal edges in column 0)
        lz2 = ['I'] * n_qubits
        for i in range(L):
            lz2[h_edge(i, 0)] = 'Z'
        
        return [''.join(lx1), ''.join(lx2)], [''.join(lz1), ''.join(lz2)]

    def qubit_coords(self) -> List[Coord2D]:
        """Return 2D coordinates for data qubits."""
        return list(self.metadata["data_coords"])

    @property
    def hx(self) -> np.ndarray:
        """X stabilizers: shape (#X-checks, #data)."""
        return self._hx

    @property
    def hz(self) -> np.ndarray:
        """Z stabilizers: shape (#Z-checks, #data)."""
        return self._hz

    @property
    def distance(self) -> int:
        """Code distance."""
        return self.metadata.get("distance", 3)

    def get_ft_gadget_config(self) -> FTGadgetCodeConfig:
        """
        Return FT gadget configuration for 2D toric codes.
        
        Toric codes have periodic boundary conditions which can cause 
        coordinate lookup issues with geometric scheduling. While we've
        added periodic boundary wrapping support, graph coloring is more
        robust for codes with non-trivial topology.
        
        Additionally, toric codes have 2 logical qubits, so gadget
        experiments need to be aware of this.
        """
        return FTGadgetCodeConfig(
            schedule_mode=ScheduleMode.GRAPH_COLORING,  # Robust for periodic BCs
            first_round_x_detectors=True,
            first_round_z_detectors=True,
            enable_metachecks=False,
        )
