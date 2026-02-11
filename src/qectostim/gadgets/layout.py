# src/qectostim/gadgets/layout.py
"""
N-Dimensional Layout Manager for Multi-Code Gadgets.

Manages spatial positioning of multiple code blocks in a shared coordinate space,
enabling gadgets to work with arbitrary combinations of 2D, 3D, and 4D codes.

Key features:
- Automatic non-overlapping block placement
- Bridge ancilla positioning for surgery/teleportation
- Local-to-global qubit index mapping
- Support for heterogeneous code dimensions (embeds lower-dim in higher-dim)
- QubitAllocation for unified experiment circuit generation
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any, Union, TYPE_CHECKING
import numpy as np
import stim

from qectostim.gadgets.coordinates import (
    CoordND,
    get_code_dimension,
    get_bounding_box,
    get_bounding_box_diagonal,
    get_centroid,
    translate_coords,
    compute_non_overlapping_offset,
    compute_bridge_position,
    pad_coord_to_dim,
    get_code_coords,
    emit_qubit_coords_nd,
)

if TYPE_CHECKING:
    from qectostim.codes.abstract_code import Code


def _get_code_stabilizer_counts(code: "Code") -> Tuple[int, int]:
    """Get (num_x_stabilizers, num_z_stabilizers) for any code type.

    Resolution order:
    1. CSS codes (``hx``/``hz`` parity-check matrices)
    2. General stabilizer codes (``stabilizer_matrix`` with CSS check)
    3. Zero fallback

    For non-CSS stabilizer codes the stabilizers cannot be cleanly split
    into X-type and Z-type.  In that case we report all stabilizers as
    "X" (i.e. ``(n_stabs, 0)``) so that the allocation reserves the
    right total number of ancillas.  The
    :class:`GeneralStabilizerRoundBuilder` uses a single unified ancilla
    pool anyway, so the split is irrelevant at runtime.

    Returns
    -------
    Tuple[int, int]
        ``(nx, nz)`` — number of X-type and Z-type stabilizers.
    """
    # 1. CSS path: hx/hz parity-check matrices
    hx_raw = getattr(code, 'hx', None)
    hz_raw = getattr(code, 'hz', None)
    if hx_raw is not None or hz_raw is not None:
        nx = hx_raw.shape[0] if hx_raw is not None and hasattr(hx_raw, 'shape') and hx_raw.size > 0 else 0
        nz = hz_raw.shape[0] if hz_raw is not None and hasattr(hz_raw, 'shape') and hz_raw.size > 0 else 0
        if nx > 0 or nz > 0:
            return nx, nz

    # 2. General stabilizer path: stabilizer_matrix
    stab_mat = getattr(code, 'stabilizer_matrix', None)
    if stab_mat is not None and hasattr(stab_mat, 'shape') and stab_mat.size > 0:
        n_stabs = stab_mat.shape[0]
        # Check if the code is actually CSS (pure X-type or pure Z-type rows)
        n = code.n
        if stab_mat.shape[1] >= 2 * n:
            x_part = stab_mat[:, :n]
            z_part = stab_mat[:, n:2*n]
            nx_count = 0
            nz_count = 0
            for i in range(n_stabs):
                has_x = x_part[i].any()
                has_z = z_part[i].any()
                if has_x and not has_z:
                    nx_count += 1
                elif has_z and not has_x:
                    nz_count += 1
                else:
                    # Mixed stabilizer — can't split; put all in x bucket
                    return n_stabs, 0
            return nx_count, nz_count
        # Fallback: all stabilizers in x bucket
        return n_stabs, 0

    # 3. Nothing found
    return 0, 0


@dataclass
class BlockAllocation:
    """Allocation info for a single code block."""
    block_name: str
    code: Any
    data_start: int
    data_count: int
    x_anc_start: int
    x_anc_count: int
    z_anc_start: int
    z_anc_count: int
    offset: CoordND = field(default_factory=lambda: (0.0, 0.0))
    
    @property
    def data_range(self) -> range:
        return range(self.data_start, self.data_start + self.data_count)
    
    @property
    def x_anc_range(self) -> range:
        return range(self.x_anc_start, self.x_anc_start + self.x_anc_count)
    
    @property
    def z_anc_range(self) -> range:
        return range(self.z_anc_start, self.z_anc_start + self.z_anc_count)
    
    @property
    def all_qubits(self) -> List[int]:
        return list(self.data_range) + list(self.x_anc_range) + list(self.z_anc_range)
    
    @property
    def total_qubits(self) -> int:
        return self.data_count + self.x_anc_count + self.z_anc_count
    
    # =========================================================================
    # Goal 7: Syndrome qubit accessors
    # =========================================================================
    
    def get_z_syndrome_qubits(self) -> List[int]:
        """Get list of Z syndrome ancilla qubit indices."""
        return list(self.z_anc_range)
    
    def get_x_syndrome_qubits(self) -> List[int]:
        """Get list of X syndrome ancilla qubit indices."""
        return list(self.x_anc_range)
    
    def get_data_qubits(self) -> List[int]:
        """Get list of data qubit indices."""
        return list(self.data_range)


@dataclass
class QubitAllocation:
    """
    Unified qubit allocation for experiment circuit generation.
    
    This replaces the ad-hoc allocation dicts used in experiments,
    providing a consistent interface for:
    - Global qubit index assignment
    - Coordinate emission
    - Block-to-global index mapping
    
    Used by FaultTolerantGadgetExperiment and all gadgets.
    """
    blocks: Dict[str, BlockAllocation] = field(default_factory=dict)
    bridge_ancillas: List[Tuple[int, CoordND, str]] = field(default_factory=list)
    _total_qubits: int = 0
    
    @classmethod
    def from_codes(
        cls,
        codes: List[Any],
        margin: float = 3.0,
    ) -> "QubitAllocation":
        """
        Create allocation from a list of codes.
        
        Assigns global indices and computes non-overlapping offsets.
        
        Parameters
        ----------
        codes : List[Code]
            List of codes to allocate.
        margin : float
            Gap between code blocks.
            
        Returns
        -------
        QubitAllocation
        """
        alloc = cls()
        idx = 0
        x_offset = 0.0
        
        for i, code in enumerate(codes):
            n = code.n
            
            # Get stabilizer counts with CSS → stabilizer_matrix fallback
            nx, nz = _get_code_stabilizer_counts(code)
            
            block_name = f"block_{i}"
            
            # Get code coordinates for offset calculation
            data_coords, _, _ = get_code_coords(code)
            if data_coords:
                _, max_corner = get_bounding_box(data_coords)
                block_width = max_corner[0] + margin if max_corner else n + margin
            else:
                block_width = n + margin
            
            block = BlockAllocation(
                block_name=block_name,
                code=code,
                data_start=idx,
                data_count=n,
                x_anc_start=idx + n,
                x_anc_count=nx,
                z_anc_start=idx + n + nx,
                z_anc_count=nz,
                offset=(x_offset, 0.0),
            )
            
            idx += n + nx + nz
            x_offset += block_width
            
            alloc.blocks[block_name] = block
        
        alloc._total_qubits = idx
        return alloc
    
    @property
    def total_qubits(self) -> int:
        """Total number of allocated qubits."""
        return self._total_qubits + len(self.bridge_ancillas)
    
    def add_bridge_ancilla(self, coord: CoordND, purpose: str = "bridge") -> int:
        """Add a bridge ancilla and return its global index."""
        global_idx = self._total_qubits + len(self.bridge_ancillas)
        self.bridge_ancillas.append((global_idx, coord, purpose))
        return global_idx
    
    def get_block(self, block_name: str) -> Optional[BlockAllocation]:
        """Get block allocation by name."""
        return self.blocks.get(block_name)
    
    def get_all_data_qubits(self) -> List[int]:
        """Get all data qubit indices across all blocks."""
        result = []
        for block in self.blocks.values():
            result.extend(block.data_range)
        return result
    
    def get_all_ancillas(self) -> List[int]:
        """Get all ancilla qubit indices (stabilizer + bridge)."""
        result = []
        for block in self.blocks.values():
            result.extend(block.x_anc_range)
            result.extend(block.z_anc_range)
        for idx, _, _ in self.bridge_ancillas:
            result.append(idx)
        return result
    
    def emit_qubit_coords(self, circuit: stim.Circuit) -> None:
        """
        Emit QUBIT_COORDS for all qubits to the circuit.
        
        Uses topology-aware placement when codes have coordinate metadata.
        """
        for block in self.blocks.values():
            code = block.code
            offset = block.offset
            
            # Get code coordinates
            data_coords, x_stab_coords, z_stab_coords = get_code_coords(code)
            
            # Data qubits
            for i, global_idx in enumerate(block.data_range):
                if i < len(data_coords):
                    coord = data_coords[i]
                    full_coord = (float(coord[0]) + offset[0], 
                                  float(coord[1]) + offset[1] if len(coord) > 1 else offset[1])
                else:
                    full_coord = (float(i) + offset[0], offset[1])
                circuit.append("QUBIT_COORDS", [global_idx], list(full_coord))
            
            # X ancillas
            for i, global_idx in enumerate(block.x_anc_range):
                if i < len(x_stab_coords):
                    coord = x_stab_coords[i]
                    full_coord = (float(coord[0]) + offset[0],
                                  float(coord[1]) + offset[1] if len(coord) > 1 else offset[1])
                else:
                    full_coord = (float(i) + offset[0], 1.0 + offset[1])
                circuit.append("QUBIT_COORDS", [global_idx], list(full_coord))
            
            # Z ancillas
            for i, global_idx in enumerate(block.z_anc_range):
                if i < len(z_stab_coords):
                    coord = z_stab_coords[i]
                    full_coord = (float(coord[0]) + offset[0],
                                  float(coord[1]) + offset[1] if len(coord) > 1 else offset[1])
                else:
                    full_coord = (float(i) + offset[0], -1.0 + offset[1])
                circuit.append("QUBIT_COORDS", [global_idx], list(full_coord))
        
        # Bridge ancillas
        for global_idx, coord, _ in self.bridge_ancillas:
            circuit.append("QUBIT_COORDS", [global_idx], list(coord))
    
    def emit_reset_all(self, circuit: stim.Circuit) -> None:
        """Emit reset for all qubits."""
        total = self.total_qubits
        if total > 0:
            circuit.append("R", list(range(total)))
            circuit.append("TICK")
    
    def local_to_global(self, block_name: str, local_idx: int, qubit_type: str = "data") -> int:
        """Convert local qubit index to global."""
        block = self.blocks.get(block_name)
        if block is None:
            raise ValueError(f"Unknown block: {block_name}")
        
        if qubit_type == "data":
            return block.data_start + local_idx
        elif qubit_type == "x_anc":
            return block.x_anc_start + local_idx
        elif qubit_type == "z_anc":
            return block.z_anc_start + local_idx
        else:
            raise ValueError(f"Unknown qubit type: {qubit_type}")


@dataclass
class BlockInfo:
    """Information about a code block in the gadget layout."""
    
    name: str
    code: Any  # Code object
    offset: CoordND  # Global offset for this block
    data_qubit_range: range  # Global indices for data qubits
    x_ancilla_range: range  # Global indices for X stabilizer ancillas
    z_ancilla_range: range  # Global indices for Z stabilizer ancillas
    local_dim: int  # Original dimension of this code's coordinates
    
    @property
    def total_qubits(self) -> int:
        """Total number of qubits (data + ancillas) in this block."""
        return (
            len(self.data_qubit_range) + 
            len(self.x_ancilla_range) + 
            len(self.z_ancilla_range)
        )


@dataclass
class BridgeAncilla:
    """Information about a bridge ancilla for surgery/teleportation."""
    
    global_idx: int
    coord: CoordND
    purpose: str  # e.g., "joint_x_measurement", "bell_pair"
    connected_blocks: List[str] = field(default_factory=list)


@dataclass
class QubitIndexMap:
    """Bidirectional mapping between local and global qubit indices."""
    
    # block_name -> {local_idx: global_idx}
    block_to_global: Dict[str, Dict[int, int]] = field(default_factory=dict)
    
    # global_idx -> (block_name, local_idx, qubit_type)
    # qubit_type: "data", "x_ancilla", "z_ancilla", "bridge"
    global_to_block: Dict[int, Tuple[str, int, str]] = field(default_factory=dict)
    
    # global_idx -> coordinate
    global_coords: Dict[int, CoordND] = field(default_factory=dict)
    
    def add_mapping(
        self, 
        block_name: str, 
        local_idx: int, 
        global_idx: int, 
        qubit_type: str,
        coord: CoordND,
    ) -> None:
        """Add a local-to-global mapping."""
        if block_name not in self.block_to_global:
            self.block_to_global[block_name] = {}
        self.block_to_global[block_name][local_idx] = global_idx
        self.global_to_block[global_idx] = (block_name, local_idx, qubit_type)
        self.global_coords[global_idx] = coord
    
    def get_global(self, block_name: str, local_idx: int) -> Optional[int]:
        """Get global index for a local qubit."""
        return self.block_to_global.get(block_name, {}).get(local_idx)
    
    def get_local(self, global_idx: int) -> Optional[Tuple[str, int, str]]:
        """Get (block_name, local_idx, qubit_type) for a global index."""
        return self.global_to_block.get(global_idx)


class GadgetLayout:
    """
    Manages spatial layout of multiple code blocks and bridge ancillas.
    
    Handles:
    - Placing code blocks at non-overlapping positions
    - Tracking local-to-global qubit index mappings
    - Positioning bridge ancillas for inter-block operations
    - Embedding lower-dimensional codes in higher-dimensional space
    
    Parameters
    ----------
    target_dim : int, optional
        Target spatial dimension. If None, inferred from codes.
        
    Examples
    --------
    >>> layout = GadgetLayout()
    >>> layout.add_block("control", surface_code_d3)
    >>> layout.add_block("target", color_code_17, auto_offset=True)
    >>> layout.add_bridge_ancilla(purpose="joint_x_measurement")
    """
    
    def __init__(self, target_dim: Optional[int] = None):
        self.target_dim = target_dim
        self.blocks: Dict[str, BlockInfo] = {}
        self.bridge_ancillas: List[BridgeAncilla] = []
        self.qubit_map = QubitIndexMap()
        self._next_global_idx = 0
    
    @property
    def dim(self) -> int:
        """Current spatial dimension of the layout."""
        if self.target_dim is not None:
            return self.target_dim
        
        # Infer from blocks
        max_dim = 2
        for block in self.blocks.values():
            max_dim = max(max_dim, block.local_dim)
        return max_dim
    
    @property
    def total_qubits(self) -> int:
        """Total number of qubits in the layout."""
        return self._next_global_idx
    
    def add_block(
        self,
        name: str,
        code: Any,
        offset: Optional[CoordND] = None,
        auto_offset: bool = True,
        margin: float = 3.0,
    ) -> range:
        """
        Add a code block to the layout.
        
        Parameters
        ----------
        name : str
            Unique name for this block (e.g., "control", "target").
        code : Code
            The code object.
        offset : CoordND, optional
            Explicit offset. If None and auto_offset=True, computed automatically.
        auto_offset : bool
            If True and offset is None, compute non-overlapping offset.
        margin : float
            Gap between blocks when auto-computing offset.
            
        Returns
        -------
        range
            Global data qubit index range for this block.
        """
        if name in self.blocks:
            raise ValueError(f"Block '{name}' already exists")
        
        # Get code properties (n is abstract on Code ABC, always available)
        local_dim = get_code_dimension(code)
        data_coords, x_stab_coords, z_stab_coords = get_code_coords(code)
        
        n_data = code.n
        
        # Get stabilizer counts with CSS → stabilizer_matrix fallback
        n_x, n_z = _get_code_stabilizer_counts(code)
        # Use coordinate-based counts as additional fallback
        if n_x == 0 and n_z == 0:
            n_x = len(x_stab_coords)
            n_z = len(z_stab_coords)
        
        # Determine target dimension
        if self.target_dim is None:
            self.target_dim = local_dim
        else:
            self.target_dim = max(self.target_dim, local_dim)
        
        # Compute offset if needed
        if offset is None:
            if auto_offset and self.blocks:
                # Collect all existing global coordinates
                existing_coords = list(self.qubit_map.global_coords.values())
                # Pad data coords to current target dim
                padded_data = [
                    pad_coord_to_dim(c, self.dim) for c in data_coords
                ] if data_coords else [(0.0,) * self.dim]
                
                offset = compute_non_overlapping_offset(
                    existing_coords, 
                    padded_data, 
                    margin=margin,
                    direction=0,  # Offset along x-axis
                )
            else:
                offset = tuple([0.0] * self.dim)
        
        # Pad offset to target dimension
        offset = pad_coord_to_dim(offset, self.dim)
        
        # Allocate global indices
        data_start = self._next_global_idx
        data_range = range(data_start, data_start + n_data)
        self._next_global_idx += n_data
        
        x_anc_start = self._next_global_idx
        x_anc_range = range(x_anc_start, x_anc_start + n_x)
        self._next_global_idx += n_x
        
        z_anc_start = self._next_global_idx
        z_anc_range = range(z_anc_start, z_anc_start + n_z)
        self._next_global_idx += n_z
        
        # Create block info
        block = BlockInfo(
            name=name,
            code=code,
            offset=offset,
            data_qubit_range=data_range,
            x_ancilla_range=x_anc_range,
            z_ancilla_range=z_anc_range,
            local_dim=local_dim,
        )
        self.blocks[name] = block
        
        # Build qubit mappings with global coordinates
        # Data qubits
        for local_idx, global_idx in enumerate(data_range):
            if local_idx < len(data_coords):
                local_coord = pad_coord_to_dim(data_coords[local_idx], self.dim)
                global_coord = tuple(
                    c + o for c, o in zip(local_coord, offset)
                )
            else:
                global_coord = offset
            
            self.qubit_map.add_mapping(name, local_idx, global_idx, "data", global_coord)
        
        # X ancillas
        for local_idx, global_idx in enumerate(x_anc_range):
            if local_idx < len(x_stab_coords):
                local_coord = pad_coord_to_dim(x_stab_coords[local_idx], self.dim)
                global_coord = tuple(
                    c + o for c, o in zip(local_coord, offset)
                )
            else:
                global_coord = offset
            
            self.qubit_map.add_mapping(
                name, n_data + local_idx, global_idx, "x_ancilla", global_coord
            )
        
        # Z ancillas  
        for local_idx, global_idx in enumerate(z_anc_range):
            if local_idx < len(z_stab_coords):
                local_coord = pad_coord_to_dim(z_stab_coords[local_idx], self.dim)
                global_coord = tuple(
                    c + o for c, o in zip(local_coord, offset)
                )
            else:
                global_coord = offset
            
            self.qubit_map.add_mapping(
                name, n_data + n_x + local_idx, global_idx, "z_ancilla", global_coord
            )
        
        return data_range

    def add_block_adjacent(
        self,
        name: str,
        code: Any,
        reference_block: str,
        my_edge: str,
        their_edge: str,
        gap: float = 0.0,
    ) -> range:
        """
        Add a code block adjacent to an existing block with aligned boundaries.

        This enables true lattice surgery where patches share a boundary
        rather than using bridge ancillas between separated patches.

        Parameters
        ----------
        name : str
            Unique name for this block.
        code : Code
            The code object.
        reference_block : str
            Name of the existing block to position relative to.
        my_edge : str
            Which edge of the NEW block to align: "left", "right", "top", "bottom".
        their_edge : str
            Which edge of the REFERENCE block to align to.
        gap : float
            Gap between boundaries. 0 = touching, >0 = surgery channel.

        Returns
        -------
        range
            Global data qubit index range for this block.

        Raises
        ------
        ValueError
            If reference_block doesn't exist.

        Examples
        --------
        >>> layout.add_block("block_0", code, offset=(0, 0))
        >>> # Place block_1 so its LEFT edge touches block_0's RIGHT edge
        >>> layout.add_block_adjacent("block_1", code, "block_0", 
        ...                           my_edge="left", their_edge="right", gap=0)
        """
        if reference_block not in self.blocks:
            raise ValueError(f"Reference block '{reference_block}' does not exist")

        ref_block = self.blocks[reference_block]
        ref_coords = []
        for gidx in ref_block.data_qubit_range:
            if gidx in self.qubit_map.global_coords:
                ref_coords.append(self.qubit_map.global_coords[gidx])

        if not ref_coords:
            # Fall back to standard auto-offset
            return self.add_block(name, code, auto_offset=True)

        # Get bounding box of reference block
        ref_min, ref_max = get_bounding_box(ref_coords)

        # Get new block's local coordinates and bounding box
        local_dim = get_code_dimension(code)
        data_coords, x_stab_coords, z_stab_coords = get_code_coords(code)
        if not data_coords:
            return self.add_block(name, code, auto_offset=True)

        # Pad to current dim
        data_coords_padded = [pad_coord_to_dim(c, self.dim) for c in data_coords]
        new_min, new_max = get_bounding_box(data_coords_padded)

        # Compute offset based on edge alignment
        offset = [0.0] * self.dim

        # Horizontal alignment (left/right edges)
        if their_edge == "right" and my_edge == "left":
            # New block's left edge at reference's right edge + gap
            offset[0] = ref_max[0] - new_min[0] + gap
        elif their_edge == "left" and my_edge == "right":
            # New block's right edge at reference's left edge - gap
            offset[0] = ref_min[0] - new_max[0] - gap
        elif their_edge == "right" and my_edge == "right":
            # Align right edges
            offset[0] = ref_max[0] - new_max[0]
        elif their_edge == "left" and my_edge == "left":
            # Align left edges
            offset[0] = ref_min[0] - new_min[0]

        # Vertical alignment (top/bottom edges)
        if their_edge == "bottom" and my_edge == "top":
            # New block's top edge at reference's bottom edge - gap
            offset[1] = ref_min[1] - new_max[1] - gap
        elif their_edge == "top" and my_edge == "bottom":
            # New block's bottom edge at reference's top edge + gap
            offset[1] = ref_max[1] - new_min[1] + gap
        elif their_edge == "top" and my_edge == "top":
            # Align top edges
            offset[1] = ref_max[1] - new_max[1]
        elif their_edge == "bottom" and my_edge == "bottom":
            # Align bottom edges
            offset[1] = ref_min[1] - new_min[1]

        # For horizontal adjacency, align Y centers
        if their_edge in ("left", "right") and my_edge in ("left", "right"):
            ref_center_y = (ref_min[1] + ref_max[1]) / 2
            new_center_y = (new_min[1] + new_max[1]) / 2
            offset[1] = ref_center_y - new_center_y

        # For vertical adjacency, align X centers
        if their_edge in ("top", "bottom") and my_edge in ("top", "bottom"):
            ref_center_x = (ref_min[0] + ref_max[0]) / 2
            new_center_x = (new_min[0] + new_max[0]) / 2
            offset[0] = ref_center_x - new_center_x

        return self.add_block(name, code, offset=tuple(offset), auto_offset=False)

    def add_bridge_ancilla(
        self,
        purpose: str,
        coord: Optional[CoordND] = None,
        connected_blocks: Optional[List[str]] = None,
    ) -> int:
        """
        Add a bridge ancilla for inter-block operations.
        
        Parameters
        ----------
        purpose : str
            Description of ancilla purpose (e.g., "joint_x_measurement").
        coord : CoordND, optional
            Explicit coordinate. If None, computed as midpoint of connected blocks.
        connected_blocks : List[str], optional
            Names of blocks this ancilla connects. If None and coord is None,
            uses all blocks.
            
        Returns
        -------
        int
            Global index of the bridge ancilla.
        """
        if connected_blocks is None:
            connected_blocks = list(self.blocks.keys())
        
        # Compute coordinate if not provided
        if coord is None and len(connected_blocks) >= 2:
            block_a = self.blocks[connected_blocks[0]]
            block_b = self.blocks[connected_blocks[1]]
            
            coords_a, _, _ = get_code_coords(block_a.code)
            coords_b, _, _ = get_code_coords(block_b.code)
            
            # Pad to current dim
            coords_a = [pad_coord_to_dim(c, self.dim) for c in coords_a]
            coords_b = [pad_coord_to_dim(c, self.dim) for c in coords_b]
            
            coord = compute_bridge_position(
                coords_a, coords_b,
                pad_coord_to_dim(block_a.offset, self.dim),
                pad_coord_to_dim(block_b.offset, self.dim),
            )
        elif coord is None:
            coord = tuple([0.0] * self.dim)
        
        # Pad coord to target dim
        coord = pad_coord_to_dim(coord, self.dim)
        
        # Allocate global index
        global_idx = self._next_global_idx
        self._next_global_idx += 1
        
        # Create bridge ancilla
        bridge = BridgeAncilla(
            global_idx=global_idx,
            coord=coord,
            purpose=purpose,
            connected_blocks=connected_blocks,
        )
        self.bridge_ancillas.append(bridge)
        
        # Add to qubit map
        self.qubit_map.global_to_block[global_idx] = ("_bridge", len(self.bridge_ancillas) - 1, "bridge")
        self.qubit_map.global_coords[global_idx] = coord
        
        return global_idx

    def add_boundary_bridges(
        self,
        block_a: str,
        block_b: str,
        edge_a: str,
        edge_b: str,
        purpose: str,
        logical_idx: int = 0,
    ) -> List[int]:
        """
        Add bridge ancillas along the shared boundary between two adjacent blocks.
        
        For true lattice surgery, we need ancillas that mediate measurements across
        the boundary. This places bridge ancillas at the midpoint between corresponding
        boundary qubits on each block.
        
        Parameters
        ----------
        block_a : str
            Name of first block.
        block_b : str
            Name of second block.
        edge_a : str
            Edge of block_a at the boundary ("left", "right", "top", "bottom").
        edge_b : str
            Edge of block_b at the boundary (should be opposite of edge_a).
        purpose : str
            Purpose of these bridges (e.g., "zz_merge", "xx_merge").
        logical_idx : int
            Which logical qubit's boundary to use (default 0).
            
        Returns
        -------
        List[int]
            Global indices of the created bridge ancillas.
        """
        alloc_a = self.blocks.get(block_a)
        alloc_b = self.blocks.get(block_b)
        if alloc_a is None or alloc_b is None:
            return []
        
        code_a = alloc_a.code
        code_b = alloc_b.code
        offset_a = pad_coord_to_dim(alloc_a.offset, self.dim)
        offset_b = pad_coord_to_dim(alloc_b.offset, self.dim)
        
        # Get boundary coordinates from each code
        coords_a = code_a.get_boundary_coords(edge_a, logical_idx)
        coords_b = code_b.get_boundary_coords(edge_b, logical_idx)
        
        # Apply block offsets
        coords_a_global = [
            tuple(c[i] + offset_a[i] if i < len(c) else offset_a[i] for i in range(self.dim))
            for c in coords_a
        ]
        coords_b_global = [
            tuple(c[i] + offset_b[i] if i < len(c) else offset_b[i] for i in range(self.dim))
            for c in coords_b
        ]
        
        bridge_indices = []
        
        # Create bridges at midpoints between corresponding boundary qubits
        # If boundaries have different sizes, use the smaller set
        n_bridges = min(len(coords_a_global), len(coords_b_global))
        
        if n_bridges == 0:
            # Fallback: create a single bridge at the midpoint between block centroids
            idx = self.add_bridge_ancilla(
                purpose=purpose,
                connected_blocks=[block_a, block_b]
            )
            return [idx]
        
        # Sort coordinates to match them up
        # For horizontal edges (left/right), sort by Y coordinate
        # For vertical edges (top/bottom), sort by X coordinate
        if edge_a in ("left", "right"):
            coords_a_global = sorted(coords_a_global, key=lambda c: c[1] if len(c) > 1 else 0)
            coords_b_global = sorted(coords_b_global, key=lambda c: c[1] if len(c) > 1 else 0)
        else:
            coords_a_global = sorted(coords_a_global, key=lambda c: c[0])
            coords_b_global = sorted(coords_b_global, key=lambda c: c[0])
        
        for i in range(n_bridges):
            ca = coords_a_global[i]
            cb = coords_b_global[i]
            # Midpoint between the two boundary qubits
            midpoint = tuple((ca[j] + cb[j]) / 2 for j in range(self.dim))
            
            idx = self.add_bridge_ancilla(
                purpose=f"{purpose}_{i}",
                coord=midpoint,
                connected_blocks=[block_a, block_b]
            )
            bridge_indices.append(idx)
        
        return bridge_indices

    def get_boundary_qubit_pairs(
        self,
        block_a: str,
        block_b: str,
        edge_a: str,
        edge_b: str,
        logical_idx: int = 0,
    ) -> List[Tuple[int, int]]:
        """
        Get pairs of (global_idx_a, global_idx_b) for boundary qubits.
        
        These are the data qubit pairs that should be involved in a merge operation.
        
        Parameters
        ----------
        block_a, block_b : str
            Names of the two blocks.
        edge_a, edge_b : str
            Edges at the shared boundary.
        logical_idx : int
            Which logical qubit's boundary to use.
            
        Returns
        -------
        List[Tuple[int, int]]
            Pairs of global qubit indices, one from each block.
        """
        alloc_a = self.blocks.get(block_a)
        alloc_b = self.blocks.get(block_b)
        if alloc_a is None or alloc_b is None:
            return []
        
        code_a = alloc_a.code
        code_b = alloc_b.code
        
        # Get boundary qubits (local indices)
        local_a = code_a.get_boundary_qubits(edge_a, logical_idx)
        local_b = code_b.get_boundary_qubits(edge_b, logical_idx)
        
        # Get their coordinates for matching
        coords_a = code_a.get_boundary_coords(edge_a, logical_idx)
        coords_b = code_b.get_boundary_coords(edge_b, logical_idx)
        
        # Convert to global indices
        global_a = [self.local_to_global(block_a, l) for l in local_a]
        global_b = [self.local_to_global(block_b, l) for l in local_b]
        
        # Create (coord, global_idx) pairs and sort
        pairs_a = list(zip(coords_a, global_a))
        pairs_b = list(zip(coords_b, global_b))
        
        # Sort by perpendicular coordinate
        if edge_a in ("left", "right"):
            pairs_a = sorted(pairs_a, key=lambda p: p[0][1] if len(p[0]) > 1 else 0)
            pairs_b = sorted(pairs_b, key=lambda p: p[0][1] if len(p[0]) > 1 else 0)
        else:
            pairs_a = sorted(pairs_a, key=lambda p: p[0][0])
            pairs_b = sorted(pairs_b, key=lambda p: p[0][0])
        
        # Pair up
        n = min(len(pairs_a), len(pairs_b))
        result = []
        for i in range(n):
            ga = pairs_a[i][1]
            gb = pairs_b[i][1]
            if ga is not None and gb is not None:
                result.append((ga, gb))
        
        return result
    
    def get_block_data_qubits(self, block_name: str) -> List[int]:
        """Get global indices of data qubits for a block."""
        block = self.blocks.get(block_name)
        if block is None:
            return []
        return list(block.data_qubit_range)
    
    def get_block_x_ancillas(self, block_name: str) -> List[int]:
        """Get global indices of X-stabilizer ancillas for a block."""
        block = self.blocks.get(block_name)
        if block is None:
            return []
        return list(block.x_ancilla_range)
    
    def get_block_z_ancillas(self, block_name: str) -> List[int]:
        """Get global indices of Z-stabilizer ancillas for a block."""
        block = self.blocks.get(block_name)
        if block is None:
            return []
        return list(block.z_ancilla_range)
    
    def get_all_data_qubits(self) -> List[int]:
        """Get global indices of all data qubits across all blocks."""
        result = []
        for block in self.blocks.values():
            result.extend(block.data_qubit_range)
        return result
    
    def get_all_ancillas(self) -> List[int]:
        """Get global indices of all ancillas (stabilizer + bridge)."""
        result = []
        for block in self.blocks.values():
            result.extend(block.x_ancilla_range)
            result.extend(block.z_ancilla_range)
        for bridge in self.bridge_ancillas:
            result.append(bridge.global_idx)
        return result
    
    def get_coord(self, global_idx: int) -> CoordND:
        """Get coordinate for a global qubit index."""
        return self.qubit_map.global_coords.get(global_idx, tuple([0.0] * self.dim))
    
    def local_to_global(self, block_name: str, local_idx: int) -> Optional[int]:
        """Convert local qubit index to global."""
        return self.qubit_map.get_global(block_name, local_idx)
    
    def global_to_local(self, global_idx: int) -> Optional[Tuple[str, int, str]]:
        """Convert global qubit index to (block_name, local_idx, qubit_type)."""
        return self.qubit_map.get_local(global_idx)

    def get_qubit_index_map(self) -> QubitIndexMap:
        """
        Get the qubit index mapping object.
        
        Returns
        -------
        QubitIndexMap
            The mapping between local and global qubit indices.
        """
        return self.qubit_map

    def allocate_qubits(self) -> QubitAllocation:
        """
        Convert this layout to a QubitAllocation object.
        
        Creates a QubitAllocation with all the blocks and their
        qubit allocations, suitable for use by experiment classes.
        
        Returns
        -------
        QubitAllocation
            Qubit allocation suitable for circuit generation.
        """
        alloc = QubitAllocation()
        
        for name, block_info in self.blocks.items():
            block = BlockAllocation(
                block_name=name,
                code=block_info.code,
                data_start=block_info.data_qubit_range.start,
                data_count=len(block_info.data_qubit_range),
                x_anc_start=block_info.x_ancilla_range.start,
                x_anc_count=len(block_info.x_ancilla_range),
                z_anc_start=block_info.z_ancilla_range.start,
                z_anc_count=len(block_info.z_ancilla_range),
                offset=block_info.offset,
            )
            alloc.blocks[name] = block
        
        # Add bridge ancillas
        for bridge in self.bridge_ancillas:
            alloc.bridge_ancillas.append(
                (bridge.global_idx, bridge.coord, bridge.purpose)
            )
        
        alloc._total_qubits = self.total_qubits
        
        return alloc
    
    def get_qubit_coords(self) -> Dict[int, CoordND]:
        """
        Get mapping from global qubit indices to coordinates.
        
        Returns
        -------
        Dict[int, CoordND]
            Mapping from qubit index to spatial coordinates.
        """
        return dict(self.qubit_map.global_coords)
