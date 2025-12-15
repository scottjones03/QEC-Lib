# src/qectostim/gadgets/coordinates.py
"""
N-Dimensional Coordinate Utilities for Gadgets.

Provides utilities for handling coordinates across 2D, 3D, 4D (and higher) 
topological codes, enabling gadget circuits to work with arbitrary dimensions.

Key features:
- CoordND type supporting arbitrary spatial dimensions
- Bounding box computation for any-dimensional coordinate sets
- Bridge ancilla positioning between code blocks
- Stim circuit emission with proper QUBIT_COORDS and DETECTOR coordinates
"""
from __future__ import annotations

from typing import List, Tuple, Optional, Union, Dict, Any
import numpy as np
import stim

# Type aliases for coordinates
Coord2D = Tuple[float, float]
Coord3D = Tuple[float, float, float]
Coord4D = Tuple[float, float, float, float]
CoordND = Tuple[float, ...]  # Arbitrary dimension


def get_code_dimension(code: Any) -> int:
    """
    Infer spatial dimension from a code's coordinate metadata.
    
    Parameters
    ----------
    code : Code
        A code object, possibly with qubit_coords() method.
        
    Returns
    -------
    int
        Spatial dimension (2, 3, 4, etc.). Defaults to 2 if no coords found.
    """
    if hasattr(code, 'qubit_coords'):
        coords = code.qubit_coords()
        if coords and len(coords) > 0:
            first_coord = coords[0]
            if isinstance(first_coord, (list, tuple)):
                return len(first_coord)
    
    # Check metadata fallback
    if hasattr(code, '_metadata'):
        meta = code._metadata or {}
        data_coords = meta.get('data_coords', [])
        if data_coords and len(data_coords) > 0:
            return len(data_coords[0])
    
    return 2  # Default to 2D


def get_bounding_box(coords: List[CoordND]) -> Tuple[CoordND, CoordND]:
    """
    Compute the axis-aligned bounding box of a coordinate set.
    
    Parameters
    ----------
    coords : List[CoordND]
        List of N-dimensional coordinates.
        
    Returns
    -------
    Tuple[CoordND, CoordND]
        (min_corner, max_corner) of the bounding box.
        
    Examples
    --------
    >>> get_bounding_box([(0, 0), (2, 3), (1, 1)])
    ((0.0, 0.0), (2.0, 3.0))
    """
    if not coords:
        return ((), ())
    
    arr = np.array(coords, dtype=float)
    min_corner = tuple(arr.min(axis=0).tolist())
    max_corner = tuple(arr.max(axis=0).tolist())
    return min_corner, max_corner


def get_bounding_box_diagonal(coords: List[CoordND]) -> float:
    """
    Compute the diagonal length of the bounding box.
    
    Parameters
    ----------
    coords : List[CoordND]
        List of N-dimensional coordinates.
        
    Returns
    -------
    float
        Euclidean length of the bounding box diagonal.
    """
    if not coords:
        return 0.0
    
    min_corner, max_corner = get_bounding_box(coords)
    return np.linalg.norm(np.array(max_corner) - np.array(min_corner))


def get_centroid(coords: List[CoordND]) -> CoordND:
    """
    Compute the centroid (center of mass) of a coordinate set.
    
    Parameters
    ----------
    coords : List[CoordND]
        List of N-dimensional coordinates.
        
    Returns
    -------
    CoordND
        The centroid coordinate.
    """
    if not coords:
        return ()
    
    arr = np.array(coords, dtype=float)
    centroid = arr.mean(axis=0)
    return tuple(centroid.tolist())


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
        Offset to add to each coordinate.
        
    Returns
    -------
    List[CoordND]
        Translated coordinates.
    """
    if not coords:
        return []
    
    offset_arr = np.array(offset, dtype=float)
    return [tuple((np.array(c) + offset_arr).tolist()) for c in coords]


def compute_non_overlapping_offset(
    existing_coords: List[CoordND],
    new_coords: List[CoordND],
    margin: float = 2.0,
    direction: int = 0,  # 0=x, 1=y, 2=z, etc.
) -> CoordND:
    """
    Compute offset for new_coords to place them next to existing_coords without overlap.
    
    Parameters
    ----------
    existing_coords : List[CoordND]
        Coordinates already placed.
    new_coords : List[CoordND]
        Coordinates to be placed.
    margin : float
        Gap between bounding boxes.
    direction : int
        Which axis to offset along (0=x, 1=y, 2=z, ...).
        
    Returns
    -------
    CoordND
        Offset to apply to new_coords.
    """
    if not existing_coords or not new_coords:
        dim = len(new_coords[0]) if new_coords else 2
        return tuple([0.0] * dim)
    
    dim = len(existing_coords[0])
    
    # Get bounding boxes
    exist_min, exist_max = get_bounding_box(existing_coords)
    new_min, new_max = get_bounding_box(new_coords)
    
    # Compute offset so new block is to the "right" of existing along direction axis
    offset = [0.0] * dim
    
    # New block's min should be at existing block's max + margin
    shift = exist_max[direction] - new_min[direction] + margin
    offset[direction] = shift
    
    return tuple(offset)


def compute_bridge_position(
    block_a_coords: List[CoordND],
    block_b_coords: List[CoordND],
    offset_a: CoordND,
    offset_b: CoordND,
) -> CoordND:
    """
    Compute position for bridge ancilla(s) between two code blocks.
    
    Places the bridge at the midpoint between the two block centroids.
    
    Parameters
    ----------
    block_a_coords : List[CoordND]
        Local coordinates of block A.
    block_b_coords : List[CoordND]
        Local coordinates of block B.
    offset_a : CoordND
        Global offset of block A.
    offset_b : CoordND
        Global offset of block B.
        
    Returns
    -------
    CoordND
        Global coordinate for bridge ancilla.
    """
    # Get global centroids
    centroid_a = get_centroid(translate_coords(block_a_coords, offset_a))
    centroid_b = get_centroid(translate_coords(block_b_coords, offset_b))
    
    if not centroid_a or not centroid_b:
        return ()
    
    # Midpoint
    midpoint = tuple(
        (a + b) / 2.0 
        for a, b in zip(centroid_a, centroid_b)
    )
    return midpoint


def pad_coord_to_dim(coord: CoordND, target_dim: int, pad_value: float = 0.0) -> CoordND:
    """
    Pad a coordinate to a target dimension.
    
    Useful for embedding lower-dimensional codes in higher-dimensional spaces.
    
    Parameters
    ----------
    coord : CoordND
        Original coordinate.
    target_dim : int
        Target number of dimensions.
    pad_value : float
        Value to use for padding (default 0.0).
        
    Returns
    -------
    CoordND
        Padded coordinate.
        
    Examples
    --------
    >>> pad_coord_to_dim((1.0, 2.0), 4)
    (1.0, 2.0, 0.0, 0.0)
    """
    current_dim = len(coord)
    if current_dim >= target_dim:
        return coord
    
    return coord + tuple([pad_value] * (target_dim - current_dim))


def emit_qubit_coords_nd(
    circuit: stim.Circuit,
    qubit_idx: int,
    coord: CoordND,
) -> None:
    """
    Emit QUBIT_COORDS instruction for a qubit with N-dimensional coordinates.
    
    Stim supports arbitrary coordinate dimensions.
    
    Parameters
    ----------
    circuit : stim.Circuit
        Circuit to append to.
    qubit_idx : int
        Qubit index.
    coord : CoordND
        Spatial coordinates (any dimension).
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
    Emit DETECTOR instruction with spatial coordinates plus time.
    
    The time dimension is appended as the final coordinate.
    
    Parameters
    ----------
    circuit : stim.Circuit
        Circuit to append to.
    rec_indices : List[int]
        Measurement record indices (absolute).
    coord : CoordND
        Spatial coordinates.
    time : float
        Time coordinate (appended as final dimension).
    m_index : int
        Current measurement index for computing lookbacks.
    """
    if not rec_indices:
        return
    
    # Compute lookbacks from current measurement index
    lookbacks = [idx - m_index for idx in rec_indices]
    targets = [stim.target_rec(lb) for lb in lookbacks]
    
    # Append time as final coordinate
    full_coords = list(coord) + [time]
    circuit.append("DETECTOR", targets, full_coords)


def emit_observable_include(
    circuit: stim.Circuit,
    rec_indices: List[int],
    observable_idx: int,
    m_index: int,
) -> None:
    """
    Emit OBSERVABLE_INCLUDE instruction.
    
    Parameters
    ----------
    circuit : stim.Circuit
        Circuit to append to.
    rec_indices : List[int]
        Measurement record indices (absolute).
    observable_idx : int
        Which logical observable (0, 1, ...).
    m_index : int
        Current measurement index for computing lookbacks.
    """
    if not rec_indices:
        return
    
    lookbacks = [idx - m_index for idx in rec_indices]
    targets = [stim.target_rec(lb) for lb in lookbacks]
    circuit.append("OBSERVABLE_INCLUDE", targets, observable_idx)


def get_code_coords(code: Any) -> Tuple[List[CoordND], List[CoordND], List[CoordND]]:
    """
    Extract data, X-stabilizer, and Z-stabilizer coordinates from a code.
    
    Parameters
    ----------
    code : Code
        A topological code with coordinate metadata.
        
    Returns
    -------
    Tuple[List[CoordND], List[CoordND], List[CoordND]]
        (data_coords, x_stab_coords, z_stab_coords)
    """
    data_coords: List[CoordND] = []
    x_stab_coords: List[CoordND] = []
    z_stab_coords: List[CoordND] = []
    
    # Try qubit_coords() method first - but handle None return
    if hasattr(code, 'qubit_coords'):
        coords = code.qubit_coords()
        if coords is not None:
            data_coords = [tuple(c) for c in coords]
    
    # Get stabilizer coords from metadata
    if hasattr(code, '_metadata'):
        meta = code._metadata or {}
        if 'data_coords' in meta and not data_coords:
            data_coords = [tuple(c) for c in meta['data_coords']]
        if 'x_stab_coords' in meta:
            x_stab_coords = [tuple(c) for c in meta['x_stab_coords']]
        if 'z_stab_coords' in meta:
            z_stab_coords = [tuple(c) for c in meta['z_stab_coords']]
    
    return data_coords, x_stab_coords, z_stab_coords
