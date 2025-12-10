# src/qectostim/gadgets/coordinates.py
"""
N-Dimensional Coordinate Utilities for Gadgets.

This module provides utilities for handling coordinates of arbitrary dimension
(2D, 3D, 4D, or higher) across gadget circuits. It supports:
- Coordinate type inference from code metadata
- Bounding box computation for layout
- Bridge ancilla positioning between code blocks
- Stim circuit emission for QUBIT_COORDS and DETECTOR with N-D coordinates
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union, Sequence
from dataclasses import dataclass
import numpy as np

import stim

from qectostim.codes.abstract_code import Code


# Type alias for N-dimensional coordinates (supports 2D, 3D, 4D, etc.)
CoordND = Tuple[float, ...]

# Convenience aliases
Coord2D = Tuple[float, float]
Coord3D = Tuple[float, float, float]
Coord4D = Tuple[float, float, float, float]


def get_code_dimension(code: Code) -> int:
    """
    Infer spatial dimension from code's coordinate metadata.
    
    Parameters
    ----------
    code : Code
        A code object that may have qubit_coords() method.
        
    Returns
    -------
    int
        Spatial dimension (2, 3, 4, etc.). Defaults to 2 if no coords found.
    """
    if hasattr(code, 'qubit_coords'):
        coords = code.qubit_coords()
        if coords and len(coords) > 0:
            first_coord = coords[0]
            if isinstance(first_coord, (tuple, list)):
                return len(first_coord)
    
    # Check metadata for coordinate info
    meta = getattr(code, '_metadata', {}) or {}
    data_coords = meta.get('data_coords', [])
    if data_coords and len(data_coords) > 0:
        first_coord = data_coords[0]
        if isinstance(first_coord, (tuple, list)):
            return len(first_coord)
    
    return 2  # Default to 2D


def normalize_coord(coord: Sequence[float], target_dim: int) -> CoordND:
    """
    Normalize a coordinate to target dimension by padding with zeros.
    
    Parameters
    ----------
    coord : Sequence[float]
        Original coordinate of any dimension.
    target_dim : int
        Target dimension to normalize to.
        
    Returns
    -------
    CoordND
        Coordinate padded to target dimension.
        
    Examples
    --------
    >>> normalize_coord((1.0, 2.0), 3)
    (1.0, 2.0, 0.0)
    >>> normalize_coord((1.0, 2.0, 3.0, 4.0), 3)
    (1.0, 2.0, 3.0)  # Truncates if larger
    """
    coord_list = list(coord)
    if len(coord_list) < target_dim:
        coord_list.extend([0.0] * (target_dim - len(coord_list)))
    elif len(coord_list) > target_dim:
        coord_list = coord_list[:target_dim]
    return tuple(coord_list)


def get_bounding_box(coords: List[CoordND]) -> Tuple[CoordND, CoordND]:
    """
    Compute bounding box of a set of N-dimensional coordinates.
    
    Parameters
    ----------
    coords : List[CoordND]
        List of coordinates, all of the same dimension.
        
    Returns
    -------
    Tuple[CoordND, CoordND]
        (min_corner, max_corner) of the bounding box.
        
    Raises
    ------
    ValueError
        If coords is empty or coordinates have inconsistent dimensions.
    """
    if not coords:
        raise ValueError("Cannot compute bounding box of empty coordinate list")
    
    dim = len(coords[0])
    arr = np.array(coords, dtype=float)
    
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    
    min_corner = tuple(arr.min(axis=0).tolist())
    max_corner = tuple(arr.max(axis=0).tolist())
    
    return min_corner, max_corner


def get_bounding_box_diagonal(coords: List[CoordND]) -> float:
    """
    Compute the diagonal length of the bounding box.
    
    Parameters
    ----------
    coords : List[CoordND]
        List of coordinates.
        
    Returns
    -------
    float
        Euclidean length of the bounding box diagonal.
    """
    if not coords:
        return 0.0
    
    min_corner, max_corner = get_bounding_box(coords)
    return np.linalg.norm(np.array(max_corner) - np.array(min_corner))


def compute_min_pairwise_distance(coords: List[CoordND]) -> float:
    """
    Compute minimum pairwise distance between coordinates.
    
    Parameters
    ----------
    coords : List[CoordND]
        List of coordinates.
        
    Returns
    -------
    float
        Minimum pairwise Euclidean distance. Returns inf if < 2 points.
    """
    if len(coords) < 2:
        return float('inf')
    
    arr = np.array(coords, dtype=float)
    min_dist = float('inf')
    
    for i in range(len(arr)):
        for j in range(i + 1, len(arr)):
            dist = np.linalg.norm(arr[i] - arr[j])
            if dist > 1e-9:
                min_dist = min(min_dist, dist)
    
    return min_dist


def translate_coords(
    coords: List[CoordND], 
    offset: CoordND
) -> List[CoordND]:
    """
    Translate all coordinates by a given offset.
    
    Parameters
    ----------
    coords : List[CoordND]
        Original coordinates.
    offset : CoordND
        Translation offset (same dimension as coords).
        
    Returns
    -------
    List[CoordND]
        Translated coordinates.
    """
    if not coords:
        return []
    
    dim = len(coords[0])
    offset_arr = np.array(offset[:dim])
    
    result = []
    for coord in coords:
        translated = tuple((np.array(coord) + offset_arr).tolist())
        result.append(translated)
    
    return result


def compute_bridge_position(
    coords_a: List[CoordND],
    coords_b: List[CoordND],
    offset_a: CoordND,
    offset_b: CoordND,
) -> CoordND:
    """
    Compute midpoint position for bridge ancilla between two blocks.
    
    Parameters
    ----------
    coords_a : List[CoordND]
        Coordinates of first code block (in local frame).
    coords_b : List[CoordND]
        Coordinates of second code block (in local frame).
    offset_a : CoordND
        Global offset of first block.
    offset_b : CoordND
        Global offset of second block.
        
    Returns
    -------
    CoordND
        Position for bridge ancilla (midpoint between block centers).
    """
    if not coords_a or not coords_b:
        # Fallback to midpoint of offsets
        return tuple((np.array(offset_a) + np.array(offset_b)) / 2)
    
    # Compute centers of each block in global coordinates
    center_a = np.mean(coords_a, axis=0) + np.array(offset_a[:len(coords_a[0])])
    center_b = np.mean(coords_b, axis=0) + np.array(offset_b[:len(coords_b[0])])
    
    # Return midpoint
    midpoint = (center_a + center_b) / 2
    return tuple(midpoint.tolist())


def emit_qubit_coords_nd(
    circuit: stim.Circuit,
    qubit_idx: int,
    coord: CoordND,
) -> None:
    """
    Emit QUBIT_COORDS instruction with N-dimensional coordinates.
    
    Stim supports arbitrary coordinate dimensions in QUBIT_COORDS.
    
    Parameters
    ----------
    circuit : stim.Circuit
        Stim circuit to append to.
    qubit_idx : int
        Qubit index.
    coord : CoordND
        N-dimensional coordinate tuple.
    """
    coord_list = [float(c) for c in coord]
    circuit.append("QUBIT_COORDS", [qubit_idx], coord_list)


def emit_detector_nd(
    circuit: stim.Circuit,
    rec_indices: List[int],
    coord: CoordND,
    time: float,
    m_index: int,
) -> None:
    """
    Emit DETECTOR instruction with (spatial..., time) coordinates.
    
    The time dimension is appended as the final coordinate.
    
    Parameters
    ----------
    circuit : stim.Circuit
        Stim circuit to append to.
    rec_indices : List[int]
        Measurement record indices (absolute).
    coord : CoordND
        Spatial coordinates.
    time : float
        Time layer for this detector.
    m_index : int
        Current measurement index for computing lookbacks.
    """
    if not rec_indices:
        return
    
    # Compute lookbacks from absolute indices
    lookbacks = [idx - m_index for idx in rec_indices]
    targets = [stim.target_rec(lb) for lb in lookbacks]
    
    # Append time as final coordinate dimension
    coord_with_time = list(coord) + [time]
    
    circuit.append("DETECTOR", targets, coord_with_time)


def get_code_coords(code: Code) -> Dict[str, List[CoordND]]:
    """
    Extract all coordinate metadata from a code.
    
    Parameters
    ----------
    code : Code
        Code object with potential coordinate metadata.
        
    Returns
    -------
    Dict[str, List[CoordND]]
        Dictionary with keys 'data', 'x_stab', 'z_stab' mapping to coordinate lists.
    """
    result = {
        'data': [],
        'x_stab': [],
        'z_stab': [],
    }
    
    # Try qubit_coords() method first
    if hasattr(code, 'qubit_coords'):
        result['data'] = list(code.qubit_coords())
    
    # Check metadata
    meta = getattr(code, '_metadata', {}) or {}
    
    if 'data_coords' in meta:
        result['data'] = list(meta['data_coords'])
    if 'x_stab_coords' in meta:
        result['x_stab'] = list(meta['x_stab_coords'])
    if 'z_stab_coords' in meta:
        result['z_stab'] = list(meta['z_stab_coords'])
    
    return result


def infer_max_dimension(codes: List[Code]) -> int:
    """
    Infer maximum spatial dimension across multiple codes.
    
    Parameters
    ----------
    codes : List[Code]
        List of code objects.
        
    Returns
    -------
    int
        Maximum dimension found (defaults to 2 if none found).
    """
    max_dim = 2
    for code in codes:
        dim = get_code_dimension(code)
        max_dim = max(max_dim, dim)
    return max_dim
