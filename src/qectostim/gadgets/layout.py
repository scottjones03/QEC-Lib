# src/qectostim/gadgets/layout.py
"""
N-Dimensional Layout Manager for Multi-Code Gadgets.

This module manages spatial layout of multiple code blocks in a shared
coordinate space, supporting gadgets that operate on 2+ codes simultaneously
(e.g., CSS surgery, teleportation).

Key features:
- Automatic non-overlapping block placement
- Bridge ancilla positioning for joint measurements
- Local-to-global qubit index mapping
- Support for arbitrary spatial dimensions (2D, 3D, 4D, ...)
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
import numpy as np

from qectostim.codes.abstract_code import Code
from qectostim.gadgets.coordinates import (
    CoordND,
    get_code_dimension,
    get_code_coords,
    get_bounding_box,
    get_bounding_box_diagonal,
    compute_min_pairwise_distance,
    translate_coords,
    normalize_coord,
    infer_max_dimension,
)


@dataclass
class BlockInfo:
    """Information about a single code block in the layout."""
    
    name: str
    code: Code
    offset: CoordND  # Global offset for this block
    data_qubit_range: range  # Global indices for data qubits
    x_ancilla_range: range  # Global indices for X stabilizer ancillas
    z_ancilla_range: range  # Global indices for Z stabilizer ancillas
    
    @property
    def total_qubits(self) -> int:
        """Total number of qubits (data + ancilla) in this block."""
        return len(self.data_qubit_range) + len(self.x_ancilla_range) + len(self.z_ancilla_range)
    
    @property
    def all_qubit_range(self) -> range:
        """Range covering all qubits in this block."""
        start = self.data_qubit_range.start
        end = max(
            self.data_qubit_range.stop if self.data_qubit_range else start,
            self.x_ancilla_range.stop if self.x_ancilla_range else start,
            self.z_ancilla_range.stop if self.z_ancilla_range else start,
        )
        return range(start, end)


@dataclass
class BridgeAncilla:
    """Information about a bridge ancilla between code blocks."""
    
    global_idx: int
    coord: CoordND
    purpose: str  # e.g., 'x_merge', 'z_merge', 'bell_pair'
    connected_blocks: List[str]  # Names of blocks this ancilla connects


@dataclass
class QubitIndexMap:
    """Bidirectional mapping between local and global qubit indices."""
    
    # block_name -> {local_idx: global_idx}
    block_to_global: Dict[str, Dict[int, int]] = field(default_factory=dict)
    
    # global_idx -> (block_name, local_idx)
    global_to_block: Dict[int, Tuple[str, int]] = field(default_factory=dict)
    
    # global_idx -> coordinate
    global_coords: Dict[int, CoordND] = field(default_factory=dict)
    
    def add_mapping(
        self, 
        block_name: str, 
        local_idx: int, 
        global_idx: int, 
        coord: CoordND
    ) -> None:
        """Add a qubit mapping."""
        if block_name not in self.block_to_global:
            self.block_to_global[block_name] = {}
        
        self.block_to_global[block_name][local_idx] = global_idx
        self.global_to_block[global_idx] = (block_name, local_idx)
        self.global_coords[global_idx] = coord
    
    def get_global(self, block_name: str, local_idx: int) -> Optional[int]:
        """Get global index for a local qubit."""
        return self.block_to_global.get(block_name, {}).get(local_idx)
    
    def get_local(self, global_idx: int) -> Optional[Tuple[str, int]]:
        """Get (block_name, local_idx) for a global index."""
        return self.global_to_block.get(global_idx)


class GadgetLayout:
    """
    Manages spatial layout of multiple code blocks and bridge ancillas.
    
    This class handles:
    - Positioning code blocks in a shared N-dimensional coordinate space
    - Assigning global qubit indices across all blocks
    - Tracking bridge ancillas for joint measurements
    - Providing coordinate data for Stim circuit generation
    
    Parameters
    ----------
    dim : int, optional
        Spatial dimension (2, 3, 4, etc.). If None, inferred from codes.
    """
    
    def __init__(self, dim: Optional[int] = None):
        self._dim = dim
        self._blocks: Dict[str, BlockInfo] = {}
        self._bridge_ancillas: List[BridgeAncilla] = []
        self._qubit_map = QubitIndexMap()
        self._next_qubit_idx = 0
    
    @property
    def dim(self) -> int:
        """Spatial dimension of the layout."""
        return self._dim or 2
    
    @property
    def blocks(self) -> Dict[str, BlockInfo]:
        """Dictionary of block name -> BlockInfo."""
        return self._blocks
    
    @property
    def bridge_ancillas(self) -> List[BridgeAncilla]:
        """List of bridge ancillas."""
        return self._bridge_ancillas
    
    @property
    def qubit_map(self) -> QubitIndexMap:
        """Qubit index mapping."""
        return self._qubit_map
    
    @property
    def total_qubits(self) -> int:
        """Total number of qubits across all blocks and bridge ancillas."""
        return self._next_qubit_idx
    
    def add_block(
        self, 
        name: str, 
        code: Code, 
        offset: Optional[CoordND] = None
    ) -> BlockInfo:
        """
        Add a code block to the layout.
        
        Parameters
        ----------
        name : str
            Unique name for this block.
        code : Code
            The code object.
        offset : CoordND, optional
            Global offset for the block. If None, computed automatically.
            
        Returns
        -------
        BlockInfo
            Information about the added block.
        """
        if name in self._blocks:
            raise ValueError(f"Block '{name}' already exists in layout")
        
        # Infer dimension if not set
        if self._dim is None:
            self._dim = get_code_dimension(code)
        
        # Get code coordinates
        code_coords = get_code_coords(code)
        n_data = code.n
        n_x = code.hx.shape[0] if hasattr(code, 'hx') and code.hx is not None else 0
        n_z = code.hz.shape[0] if hasattr(code, 'hz') and code.hz is not None else 0
        
        # Compute offset if not provided
        if offset is None:
            offset = self._compute_auto_offset(code_coords)
        
        # Normalize offset to layout dimension
        offset = normalize_coord(offset, self.dim)
        
        # Allocate global qubit indices
        data_start = self._next_qubit_idx
        data_range = range(data_start, data_start + n_data)
        self._next_qubit_idx += n_data
        
        x_anc_start = self._next_qubit_idx
        x_anc_range = range(x_anc_start, x_anc_start + n_x)
        self._next_qubit_idx += n_x
        
        z_anc_start = self._next_qubit_idx
        z_anc_range = range(z_anc_start, z_anc_start + n_z)
        self._next_qubit_idx += n_z
        
        # Create block info
        block = BlockInfo(
            name=name,
            code=code,
            offset=offset,
            data_qubit_range=data_range,
            x_ancilla_range=x_anc_range,
            z_ancilla_range=z_anc_range,
        )
        self._blocks[name] = block
        
        # Register qubit mappings
        self._register_block_qubits(block, code_coords, offset)
        
        return block
    
    def _compute_auto_offset(self, code_coords: Dict[str, List[CoordND]]) -> CoordND:
        """Compute automatic offset to avoid overlap with existing blocks."""
        if not self._blocks:
            # First block at origin
            return tuple([0.0] * self.dim)
        
        # Get bounding box of all existing blocks
        all_coords = []
        for block in self._blocks.values():
            for global_idx in block.all_qubit_range:
                if global_idx in self._qubit_map.global_coords:
                    all_coords.append(self._qubit_map.global_coords[global_idx])
        
        if not all_coords:
            return tuple([0.0] * self.dim)
        
        # Place new block to the right of existing blocks (along first axis)
        _, max_corner = get_bounding_box(all_coords)
        
        # Get new block's extent
        new_coords = code_coords.get('data', [])
        if new_coords:
            new_diag = get_bounding_box_diagonal(new_coords)
            margin = max(new_diag * 0.5, 5.0)
        else:
            margin = 10.0
        
        # Offset along x-axis
        offset = list(max_corner)
        offset[0] += margin
        for i in range(1, len(offset)):
            offset[i] = 0.0  # Center on other axes
        
        return tuple(offset)
    
    def _register_block_qubits(
        self, 
        block: BlockInfo, 
        code_coords: Dict[str, List[CoordND]],
        offset: CoordND,
    ) -> None:
        """Register all qubits of a block in the index map."""
        # Data qubits
        data_coords = code_coords.get('data', [])
        for local_idx, global_idx in enumerate(block.data_qubit_range):
            if local_idx < len(data_coords):
                coord = normalize_coord(data_coords[local_idx], self.dim)
                global_coord = tuple((np.array(coord) + np.array(offset)).tolist())
            else:
                global_coord = offset
            
            self._qubit_map.add_mapping(block.name, local_idx, global_idx, global_coord)
        
        # X ancilla qubits
        x_coords = code_coords.get('x_stab', [])
        for i, global_idx in enumerate(block.x_ancilla_range):
            local_idx = block.code.n + i  # Local index after data qubits
            if i < len(x_coords):
                coord = normalize_coord(x_coords[i], self.dim)
                global_coord = tuple((np.array(coord) + np.array(offset)).tolist())
            else:
                global_coord = offset
            
            self._qubit_map.add_mapping(block.name, local_idx, global_idx, global_coord)
        
        # Z ancilla qubits
        z_coords = code_coords.get('z_stab', [])
        n_x = len(block.x_ancilla_range)
        for i, global_idx in enumerate(block.z_ancilla_range):
            local_idx = block.code.n + n_x + i
            if i < len(z_coords):
                coord = normalize_coord(z_coords[i], self.dim)
                global_coord = tuple((np.array(coord) + np.array(offset)).tolist())
            else:
                global_coord = offset
            
            self._qubit_map.add_mapping(block.name, local_idx, global_idx, global_coord)
    
    def add_bridge_ancilla(
        self, 
        coord: CoordND, 
        purpose: str,
        connected_blocks: Optional[List[str]] = None,
    ) -> int:
        """
        Add a bridge ancilla for joint measurements between blocks.
        
        Parameters
        ----------
        coord : CoordND
            Coordinate for the ancilla.
        purpose : str
            Description of the ancilla's purpose (e.g., 'x_merge', 'z_merge').
        connected_blocks : List[str], optional
            Names of blocks this ancilla connects.
            
        Returns
        -------
        int
            Global qubit index of the new ancilla.
        """
        coord = normalize_coord(coord, self.dim)
        global_idx = self._next_qubit_idx
        self._next_qubit_idx += 1
        
        ancilla = BridgeAncilla(
            global_idx=global_idx,
            coord=coord,
            purpose=purpose,
            connected_blocks=connected_blocks or [],
        )
        self._bridge_ancillas.append(ancilla)
        
        # Add to coordinate map (with special block name)
        self._qubit_map.global_coords[global_idx] = coord
        self._qubit_map.global_to_block[global_idx] = ('_bridge', len(self._bridge_ancillas) - 1)
        
        return global_idx
    
    def get_global_coords(self, block_name: str) -> List[CoordND]:
        """
        Get global coordinates for all data qubits in a block.
        
        Parameters
        ----------
        block_name : str
            Name of the block.
            
        Returns
        -------
        List[CoordND]
            List of global coordinates for data qubits.
        """
        if block_name not in self._blocks:
            raise ValueError(f"Block '{block_name}' not found")
        
        block = self._blocks[block_name]
        coords = []
        for global_idx in block.data_qubit_range:
            coords.append(self._qubit_map.global_coords.get(global_idx, block.offset))
        return coords
    
    def get_all_coords(self) -> Dict[int, CoordND]:
        """
        Get all global qubit coordinates.
        
        Returns
        -------
        Dict[int, CoordND]
            Mapping from global qubit index to coordinate.
        """
        return dict(self._qubit_map.global_coords)
    
    @classmethod
    def compute_non_overlapping_offsets(
        cls,
        codes: List[Code],
        arrangement: str = "linear",
        margin_factor: float = 1.5,
    ) -> List[CoordND]:
        """
        Compute non-overlapping offsets for multiple codes.
        
        Parameters
        ----------
        codes : List[Code]
            List of codes to arrange.
        arrangement : str
            Layout arrangement: 'linear', 'grid', 'vertical'.
        margin_factor : float
            Multiplier for spacing between blocks.
            
        Returns
        -------
        List[CoordND]
            List of offsets, one per code.
        """
        if not codes:
            return []
        
        dim = infer_max_dimension(codes)
        offsets: List[CoordND] = []
        
        # Compute bounding box diagonal for each code
        diagonals = []
        for code in codes:
            coords = get_code_coords(code).get('data', [])
            if coords:
                diag = get_bounding_box_diagonal(coords)
            else:
                diag = 5.0  # Default size
            diagonals.append(diag)
        
        if arrangement == "linear":
            # Arrange horizontally along x-axis
            current_x = 0.0
            for i, (code, diag) in enumerate(zip(codes, diagonals)):
                offset = [current_x] + [0.0] * (dim - 1)
                offsets.append(tuple(offset))
                current_x += diag * margin_factor
        
        elif arrangement == "vertical":
            # Arrange vertically along y-axis
            current_y = 0.0
            for i, (code, diag) in enumerate(zip(codes, diagonals)):
                offset = [0.0, current_y] + [0.0] * (dim - 2)
                offsets.append(tuple(offset))
                current_y += diag * margin_factor
        
        elif arrangement == "grid":
            # Arrange in a 2D grid
            n = len(codes)
            cols = int(np.ceil(np.sqrt(n)))
            max_diag = max(diagonals) if diagonals else 5.0
            spacing = max_diag * margin_factor
            
            for i, code in enumerate(codes):
                row = i // cols
                col = i % cols
                offset = [col * spacing, row * spacing] + [0.0] * (dim - 2)
                offsets.append(tuple(offset))
        
        else:
            raise ValueError(f"Unknown arrangement: {arrangement}")
        
        return offsets
