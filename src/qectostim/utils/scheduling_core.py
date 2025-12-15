# src/qectostim/utils/scheduling_core.py
"""
Shared scheduling algorithms and utilities.

This module provides core scheduling algorithms used by both:
- GadgetScheduler (gadgets/scheduling.py)  
- StabilizerRoundBuilder (experiments/stabilizer_rounds.py)

The main algorithms:
1. Graph coloring for conflict-free CNOT scheduling
2. Geometric scheduling using code coordinate metadata
3. Code metadata caching for coordinate lookups

By consolidating these algorithms, we ensure consistency between
memory experiments and gadget circuits.
"""
from __future__ import annotations

from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
import numpy as np


@dataclass
class CodeMetadataCache:
    """
    Cached coordinate and stabilizer metadata for a code.
    
    Avoids repeated attribute lookups and coordinate list construction.
    """
    n: int
    m_x: int = 0
    m_z: int = 0
    data_coords: List[Tuple[float, ...]] = field(default_factory=list)
    x_stab_coords: List[Tuple[float, ...]] = field(default_factory=list)
    z_stab_coords: List[Tuple[float, ...]] = field(default_factory=list)
    x_schedule: Optional[List[List[int]]] = None
    z_schedule: Optional[List[List[int]]] = None
    coord_to_idx: Dict[Tuple[float, ...], int] = field(default_factory=dict)
    
    @classmethod
    def from_code(cls, code: Any) -> "CodeMetadataCache":
        """Build cache from a code object."""
        n = code.n
        
        # Get stabilizer counts
        m_x = 0
        m_z = 0
        if hasattr(code, 'hx') and code.hx is not None:
            hx = code.hx
            m_x = hx.shape[0] if hasattr(hx, 'shape') else len(hx)
        if hasattr(code, 'hz') and code.hz is not None:
            hz = code.hz
            m_z = hz.shape[0] if hasattr(hz, 'shape') else len(hz)
        
        # Get coordinates
        data_coords = []
        if hasattr(code, 'qubit_coords'):
            qc = code.qubit_coords()
            if qc:
                data_coords = [tuple(c) if hasattr(c, '__iter__') else (float(c),) for c in qc]
        
        # Stabilizer coordinates from metadata
        x_stab_coords = []
        z_stab_coords = []
        x_schedule = None
        z_schedule = None
        
        if hasattr(code, '_metadata') and code._metadata:
            meta = code._metadata
            if 'data_coords' in meta and not data_coords:
                data_coords = [tuple(c) for c in meta['data_coords']]
            if 'x_stab_coords' in meta:
                x_stab_coords = [tuple(c) for c in meta['x_stab_coords']]
            if 'z_stab_coords' in meta:
                z_stab_coords = [tuple(c) for c in meta['z_stab_coords']]
            if 'x_schedule' in meta:
                x_schedule = meta['x_schedule']
            if 'z_schedule' in meta:
                z_schedule = meta['z_schedule']
        
        # Build coordinate lookup
        coord_to_idx = {}
        for i, coord in enumerate(data_coords):
            coord_to_idx[coord] = i
        
        return cls(
            n=n,
            m_x=m_x,
            m_z=m_z,
            data_coords=data_coords,
            x_stab_coords=x_stab_coords,
            z_stab_coords=z_stab_coords,
            x_schedule=x_schedule,
            z_schedule=z_schedule,
            coord_to_idx=coord_to_idx,
        )


def graph_coloring_cnots(
    cnots: List[Tuple[int, int]],
) -> List[List[Tuple[int, int]]]:
    """
    Partition CNOT gates into parallelizable layers using graph coloring.
    
    Two CNOTs conflict if they share any qubit. This function uses a greedy
    graph coloring algorithm to assign each CNOT to a layer such that
    no two CNOTs in the same layer share a qubit.
    
    Parameters
    ----------
    cnots : List[Tuple[int, int]]
        List of (control, target) CNOT pairs.
        
    Returns
    -------
    List[List[Tuple[int, int]]]
        Layers of non-conflicting CNOTs. Each inner list can be executed in parallel.
        
    Example
    -------
    >>> cnots = [(0, 1), (2, 3), (1, 2), (3, 4)]
    >>> layers = graph_coloring_cnots(cnots)
    >>> # layers[0] might be [(0, 1), (2, 3)]  # no conflicts
    >>> # layers[1] might be [(1, 2), (3, 4)]  # no conflicts
    """
    if not cnots:
        return []
    
    layers: List[List[Tuple[int, int]]] = []
    
    for ctrl, tgt in cnots:
        placed = False
        for layer in layers:
            # Check if this CNOT conflicts with any in this layer
            conflict = False
            for c, t in layer:
                if ctrl in (c, t) or tgt in (c, t):
                    conflict = True
                    break
            if not conflict:
                layer.append((ctrl, tgt))
                placed = True
                break
        
        if not placed:
            # Need new layer
            layers.append([(ctrl, tgt)])
    
    return layers


def geometric_scheduling_cnots(
    code: Any,
    cache: CodeMetadataCache,
    stabilizer_matrix: Any,
    ancilla_offset: int,
    data_offset: int,
    basis: str,
) -> List[List[Tuple[int, int]]]:
    """
    Schedule CNOTs using geometric scheduling metadata from the code.
    
    If the code provides x_schedule or z_schedule metadata, use it for
    optimal parallelism based on the code's natural geometry.
    
    NOTE: This function is typically used by CSSStabilizerRoundBuilder which
    handles CNOT direction separately via the is_x_type parameter in 
    _emit_graph_coloring_cnots. The correct CNOT direction convention is:
    
    - X-type stabilizers: CNOT(ancilla, data) - ancilla controls
    - Z-type stabilizers: CNOT(data, ancilla) - data controls
    
    This function returns (data, ancilla) pairs which are then used by the
    caller with appropriate direction handling.
    
    Parameters
    ----------
    code : Code
        The code object (may have _metadata with schedules).
    cache : CodeMetadataCache
        Cached code metadata.
    stabilizer_matrix : np.ndarray
        The hx or hz matrix (m x n).
    ancilla_offset : int
        Base index for ancilla qubits.
    data_offset : int
        Base index for data qubits.
    basis : str
        "X" or "Z" indicating which stabilizer type.
        
    Returns
    -------
    List[List[Tuple[int, int]]]
        CNOT layers. Returns (data, ancilla) pairs - caller handles direction.
    """
    schedule = cache.x_schedule if basis == "X" else cache.z_schedule
    
    if schedule is None:
        # No geometric schedule available - fall back to graph coloring
        cnots = []
        m = stabilizer_matrix.shape[0]
        for stab_idx in range(m):
            row = stabilizer_matrix[stab_idx]
            if hasattr(row, 'toarray'):
                row = row.toarray().flatten()
            data_qubits = [i for i, v in enumerate(row) if v]
            for dq in data_qubits:
                anc = ancilla_offset + stab_idx
                data = data_offset + dq
                # Return (data, ancilla) pairs - caller handles direction
                cnots.append((data, anc))
        return graph_coloring_cnots(cnots)
    
    # Use geometric schedule
    layers: List[List[Tuple[int, int]]] = []
    m = stabilizer_matrix.shape[0]
    
    for step_qubits in schedule:
        # step_qubits is a list of data qubit indices to interact with in this step
        layer_cnots: List[Tuple[int, int]] = []
        for stab_idx in range(m):
            row = stabilizer_matrix[stab_idx]
            if hasattr(row, 'toarray'):
                row = row.toarray().flatten()
            
            for dq in step_qubits:
                if dq < len(row) and row[dq]:
                    anc = ancilla_offset + stab_idx
                    data = data_offset + dq
                    # Return (data, ancilla) pairs - caller handles direction
                    layer_cnots.append((data, anc))
        
        if layer_cnots:
            layers.append(layer_cnots)
    
    return layers


def schedule_stabilizer_cnots(
    code: Any,
    hx: Optional[Any],
    hz: Optional[Any],
    data_offset: int,
    x_anc_offset: int,
    z_anc_offset: int,
    use_geometric: bool = True,
) -> Tuple[List[List[Tuple[int, int]]], List[List[Tuple[int, int]]]]:
    """
    Schedule all CNOTs for a stabilizer measurement round.
    
    Returns separate layer lists for X and Z stabilizers.
    
    Parameters
    ----------
    code : Code
        The quantum error correcting code.
    hx, hz : np.ndarray or None
        X and Z parity check matrices.
    data_offset : int
        Base index for data qubits.
    x_anc_offset : int
        Base index for X stabilizer ancillas.
    z_anc_offset : int
        Base index for Z stabilizer ancillas.
    use_geometric : bool
        Whether to attempt geometric scheduling (default True).
        
    Returns
    -------
    Tuple[List[List[Tuple[int, int]]], List[List[Tuple[int, int]]]]
        (x_layers, z_layers) where each layer is a list of (data, ancilla) pairs.
        Both X and Z stabilizers use data→ancilla direction.
    """
    cache = CodeMetadataCache.from_code(code)
    
    x_layers: List[List[Tuple[int, int]]] = []
    z_layers: List[List[Tuple[int, int]]] = []
    
    if hx is not None and hx.shape[0] > 0:
        if use_geometric and cache.x_schedule is not None:
            x_layers = geometric_scheduling_cnots(
                code, cache, hx, x_anc_offset, data_offset, "X"
            )
        else:
            # Fallback to graph coloring - both X and Z use data→ancilla
            cnots = []
            for stab_idx in range(hx.shape[0]):
                row = hx[stab_idx]
                if hasattr(row, 'toarray'):
                    row = row.toarray().flatten()
                for dq, v in enumerate(row):
                    if v:
                        cnots.append((data_offset + dq, x_anc_offset + stab_idx))
            x_layers = graph_coloring_cnots(cnots)
    
    if hz is not None and hz.shape[0] > 0:
        if use_geometric and cache.z_schedule is not None:
            z_layers = geometric_scheduling_cnots(
                code, cache, hz, z_anc_offset, data_offset, "Z"
            )
        else:
            # Fallback to graph coloring - both X and Z use data→ancilla
            cnots = []
            for stab_idx in range(hz.shape[0]):
                row = hz[stab_idx]
                if hasattr(row, 'toarray'):
                    row = row.toarray().flatten()
                for dq, v in enumerate(row):
                    if v:
                        cnots.append((data_offset + dq, z_anc_offset + stab_idx))
            z_layers = graph_coloring_cnots(cnots)
    
    return x_layers, z_layers


def compute_circuit_depth(
    x_layers: List[List[Tuple[int, int]]],
    z_layers: List[List[Tuple[int, int]]],
    include_prep_meas: bool = True,
) -> int:
    """
    Compute the circuit depth for a stabilizer round.
    
    Parameters
    ----------
    x_layers : List[List[Tuple[int, int]]]
        CNOT layers for X stabilizers.
    z_layers : List[List[Tuple[int, int]]]
        CNOT layers for Z stabilizers.
    include_prep_meas : bool
        Whether to include reset and measurement layers (adds 2).
        
    Returns
    -------
    int
        Total circuit depth in layers.
    """
    cnot_depth = max(len(x_layers), len(z_layers))
    if include_prep_meas:
        # 1 for reset, 1 for H on X ancillas (prep), CNOTs, H on X (meas), 1 for measure
        return 2 + cnot_depth + 2
    return cnot_depth
