# src/qectostim/codes/composite/concatenated.py
"""
Concatenated Codes: Outer ∘ Inner encoding.

This module provides concatenated code constructions where each physical qubit 
of the outer code is encoded using the inner code. The total number of physical 
qubits is n_outer * n_inner.

Naming Convention
-----------------
The terms "outer" and "inner" follow the mathematical composition notation:
    concatenated = outer ∘ inner

This means:
- **outer**: The code whose logical qubits are protected. Each physical qubit
  of the outer code becomes a logical qubit of the inner code.
- **inner**: The code that provides the physical layer. Each outer physical 
  qubit is encoded into n_inner physical qubits.

For hierarchical decoding, the perspective is reversed:
- **physical_code** (= inner): The physical-level code, closest to hardware.
  This has n_inner physical qubits per block.
- **logical_code** (= outer): The logical-level code, providing the outer 
  error correction. This has n_outer blocks.

Example: Rep(3) ∘ [[4,2,2]]
    - outer = Rep(3): 3 logical qubits, provides majority vote
    - inner = [[4,2,2]]: 4 physical qubits, encodes 2 logical qubits
    - physical_code = [[4,2,2]] (4 qubits per block)
    - logical_code = Rep(3) (3 blocks for error correction)

For CSS codes, the parity check matrices are constructed as:
- Inner checks: Block-diagonal copies of inner Hx/Hz for each outer qubit
- Outer checks: Each outer stabilizer lifted through inner logical operators

Classes
-------
ConcatenatedCode
    Abstract base for concatenated codes (stores outer/inner reference).
ConcatenatedCSSCode  
    Concatenation of two CSS codes with proper stabilizer lifting.
ConcatenatedTopologicalCSSCode
    Adds geometric layout for topological codes.
"""
from __future__ import annotations

from typing import Dict, Any, Optional, List, Tuple

import numpy as np

from qectostim.codes.abstract_code import Code, PauliString
from qectostim.codes.abstract_css import TopologicalCSSCode, CSSCode
from qectostim.codes.utils import (
    lift_pauli_through_inner,
    pauli_to_symplectic,
    symplectic_to_pauli,
    kron_gf2,
    str_to_pauli,
)


def _normalize_logical_ops(ops: List[Any]) -> List[PauliString]:
    """
    Normalize logical operators to PauliString dicts.
    
    Handles both string format ("XXXIIII") and dict format ({0: 'X', 1: 'X', ...}).
    """
    result = []
    for op in ops:
        if isinstance(op, str):
            result.append(str_to_pauli(op))
        elif isinstance(op, dict):
            result.append(op)
        else:
            raise TypeError(f"Unsupported logical operator type: {type(op)}")
    return result

Coord2D = Tuple[float, float]


def _compute_optimal_scale(
    outer_coords: List[Coord2D],
    inner_coords: List[Coord2D],
    margin: float = 3.0,
) -> float:
    """
    Compute optimal scale factor to prevent inner block overlap.
    
    The scale is chosen so that when inner code blocks are placed at
    each outer qubit position, they don't overlap. This requires:
        scale * min_outer_distance > inner_bounding_box_diagonal * margin
    
    Parameters
    ----------
    outer_coords : List[Coord2D]
        Coordinates of outer code qubits.
    inner_coords : List[Coord2D]
        Coordinates of inner code qubits.
    margin : float, optional
        Safety margin multiplier (default 3.0). Larger values give more
        spacing between inner blocks. A margin of 3.0 provides good 
        visual separation for plotting.
        
    Returns
    -------
    float
        Optimal scale factor.
        
    Examples
    --------
    >>> outer = [(0,0), (1,0), (0,1), (1,1)]  # 422-like
    >>> inner = [(1,1), (3,1), (5,1), ...]     # Surface-like
    >>> _compute_optimal_scale(outer, inner)  # Returns ~17 with margin=3.0
    """
    if len(outer_coords) < 2:
        # Single outer qubit - no spacing needed
        return 1.0
    
    # Compute inner bounding box diagonal (size of inner block)
    inner_arr = np.array(inner_coords, dtype=float)
    inner_min = inner_arr.min(axis=0)
    inner_max = inner_arr.max(axis=0)
    inner_diagonal = np.linalg.norm(inner_max - inner_min)
    
    # If inner is a single point, use a small default size
    if inner_diagonal < 1e-9:
        inner_diagonal = 1.0
    
    # Find minimum pairwise distance between outer qubits
    outer_arr = np.array(outer_coords, dtype=float)
    n_outer = len(outer_coords)
    min_outer_dist = float('inf')
    
    for i in range(n_outer):
        for j in range(i + 1, n_outer):
            dist = np.linalg.norm(outer_arr[i] - outer_arr[j])
            if dist > 1e-9:  # Avoid zero distances
                min_outer_dist = min(min_outer_dist, dist)
    
    # If all outer qubits are at the same position, use default
    if min_outer_dist == float('inf'):
        return 1.0
    
    # Scale formula: we want inner blocks separated by at least inner_diagonal
    # After scaling, outer positions are multiplied by scale
    # So: scale * min_outer_dist > inner_diagonal * margin
    # => scale > inner_diagonal * margin / min_outer_dist
    optimal_scale = (inner_diagonal * margin) / min_outer_dist
    
    # Ensure minimum scale of 1.0 and round to nice number
    return max(optimal_scale, 1.0)


class ConcatenatedCode(Code):
    """
    Abstract base class for concatenated codes: outer ∘ inner.

    It just remembers the outer and inner code objects and basic sizing;
    CSS-specific stuff lives in ConcatenatedCSSCode.
    
    Naming Convention
    -----------------
    The attributes `outer` and `inner` follow mathematical notation:
    - outer: The logical-level code (its physical qubits become inner's logicals)
    - inner: The physical-level code (provides the actual physical qubits)
    
    For decoder implementations, use the aliases:
    - physical_code: Same as inner (the code at the physical level)
    - logical_code: Same as outer (the code at the logical level)
    
    Parameters
    ----------
    outer : Code
        The outer (logical-level) code whose physical qubits are encoded.
    inner : Code  
        The inner (physical-level) code used to encode each outer qubit.
    depth : int, optional
        Concatenation depth. depth=1 means single concatenation (default).
        depth=2 means (outer ∘ inner) ∘ inner, etc.
        
    Attributes
    ----------
    outer : Code
        The outer (logical-level) code.
    inner : Code
        The inner (physical-level) code.
    physical_code : Code
        Alias for inner - the physical-level code.
    logical_code : Code
        Alias for outer - the logical-level code.
    depth : int
        Concatenation depth (1 for single concatenation).
    """

    def __init__(self, outer: Code, inner: Code, depth: int = 1) -> None:
        # Don't call super().__init__ here; CSSCode / TopologicalCSSCode
        # will take care of base init.
        self.outer = outer
        self.inner = inner
        self.depth = depth
        self._n_outer = outer.n
        self._n_inner = inner.n

    @property
    def n_outer(self) -> int:
        """Number of physical qubits in the outer code."""
        return self._n_outer

    @property
    def n_inner(self) -> int:
        """Number of physical qubits in the inner code."""
        return self._n_inner
    
    @property
    def physical_code(self) -> Code:
        """
        The physical-level code (alias for inner).
        
        This is the code that provides the actual physical qubits.
        For Rep(3) ∘ [[4,2,2]], this returns the [[4,2,2]] code.
        
        Use this property in decoder implementations for clarity.
        """
        return self.inner
    
    @property
    def logical_code(self) -> Code:
        """
        The logical-level code (alias for outer).
        
        This is the code that provides the outer error correction layer.
        For Rep(3) ∘ [[4,2,2]], this returns the Rep(3) code.
        
        Use this property in decoder implementations for clarity.
        """
        return self.outer
    
    @property
    def n_blocks(self) -> int:
        """
        Number of physical blocks in the concatenation.
        
        This equals n_outer (the number of physical qubits in the outer code),
        since each outer physical qubit becomes one encoded block.
        """
        return self._n_outer
    
    @property
    def effective_n_inner(self) -> int:
        """
        Effective inner block size accounting for depth.
        
        For depth d, each outer qubit maps to n_inner^d physical qubits.
        """
        return self._n_inner ** self.depth

    @property
    def name(self) -> str:
        outer_name = getattr(self.outer, "name", type(self.outer).__name__)
        inner_name = getattr(self.inner, "name", type(self.inner).__name__)
        if self.depth == 1:
            return f"Concatenated({outer_name}, {inner_name})"
        return f"Concatenated({outer_name}, {inner_name}, depth={self.depth})"

    # =========================================================================
    # Decoder compatibility properties
    # =========================================================================
    @property
    def is_concatenated(self) -> bool:
        """True for concatenated codes."""
        return self._metadata.get("is_concatenated", True)
    
    @property
    def inner_k(self) -> int:
        """Number of logical qubits in inner code."""
        return self._metadata.get("inner_k", getattr(self.inner, "k", 1))
    
    @property
    def outer_k(self) -> int:
        """Number of logical qubits in outer code."""
        return self._metadata.get("outer_k", getattr(self.outer, "k", 1))
    
    @property
    def supports_standard_decoders(self) -> bool:
        """True if standard MWPM decoders work via CSSMemoryExperiment."""
        return self._metadata.get("supports_standard_decoders", True)
    
    @property
    def supports_hierarchical_decoder(self) -> bool:
        """True if hierarchical concatenated decoder can be used."""
        return self._metadata.get("supports_hierarchical_decoder", True)

    def outer_block_indices(self, outer_q: int) -> List[int]:
        """
        Return physical indices of the inner block encoding outer qubit q.
        
        Parameters
        ----------
        outer_q : int
            Index of outer code qubit (0 to n_outer-1).
            
        Returns
        -------
        List[int]
            List of physical qubit indices for this block.
        """
        block_size = self.effective_n_inner
        start = outer_q * block_size
        return list(range(start, start + block_size))
    

class ConcatenatedCSSCode(CSSCode, ConcatenatedCode):
    """
    Concatenated CSS code: outer ∘ inner.

    Each physical qubit of `outer` is encoded using `inner`.
    Both outer and inner must be CSS codes.
    
    Construction
    ------------
    Given outer code with Hx_out, Hz_out and inner code with Hx_in, Hz_in,
    the concatenated stabilizers are:
    
    1. Inner checks (block-diagonal): 
       - For each outer qubit q, embed inner Hx/Hz on physical qubits 
         [q*n_inner, ..., (q+1)*n_inner - 1]
    
    2. Outer checks (lifted through inner logicals):
       - For each X-check of outer code: replace each X_q with inner logical X
       - For each Z-check of outer code: replace each Z_q with inner logical Z
       
    The logical operators are the outer logicals lifted through inner logicals.
    
    Parameters
    ----------
    outer : CSSCode
        The outer CSS code.
    inner : CSSCode
        The inner CSS code (must encode at least 1 logical qubit).
    depth : int, optional
        Concatenation depth (default 1). For depth > 1, recursively applies
        the inner code.
    metadata : dict, optional
        Additional metadata.
    """

    def __init__(
        self,
        outer: CSSCode,
        inner: CSSCode,
        depth: int = 1,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        # First remember outer/inner and sizes.
        ConcatenatedCode.__init__(self, outer, inner, depth)
        
        # For depth > 1, recursively build inner^depth
        effective_inner = inner
        for _ in range(depth - 1):
            # inner ∘ inner ∘ ... (depth times)
            effective_inner = ConcatenatedCSSCode(inner, effective_inner, depth=1)
        
        n_eff_inner = effective_inner.n
        n_concat = self._n_outer * n_eff_inner
        
        # Get inner logical operators and normalize to PauliString dicts
        inner_log_x = _normalize_logical_ops(effective_inner.logical_x_ops)
        inner_log_z = _normalize_logical_ops(effective_inner.logical_z_ops)
        
        if not inner_log_x or not inner_log_z:
            raise ValueError("Inner code must have at least one logical qubit (k >= 1)")

        # --- 1. Inner checks, block-diagonal on each outer physical qubit ----
        hx_rows: List[np.ndarray] = []
        hz_rows: List[np.ndarray] = []

        for outer_q in range(self._n_outer):
            offset = outer_q * n_eff_inner

            # X-type inner checks
            for row in effective_inner.hx:
                new_row = np.zeros(n_concat, dtype=np.uint8)
                for i, bit in enumerate(row):
                    if bit:
                        new_row[offset + i] = 1
                hx_rows.append(new_row)

            # Z-type inner checks
            for row in effective_inner.hz:
                new_row = np.zeros(n_concat, dtype=np.uint8)
                for i, bit in enumerate(row):
                    if bit:
                        new_row[offset + i] = 1
                hz_rows.append(new_row)

        # --- 2. Outer checks lifted via inner logicals -----------------------
        # IMPORTANT: Two interpretations of outer stabilizers:
        #
        # (A) FLAT MODEL (used here for hx/hz matrices):
        #     Outer X-stabilizer on qubits {k: outer_hx[j,k]=1} is lifted by
        #     replacing each X_k with inner logical X_L. This gives a stabilizer
        #     supported only on the inner logical X support within each block.
        #     This is the correct stabilizer for the concatenated code as a 
        #     single large CSS code.
        #
        # (B) HIERARCHICAL MODEL (for FT measurement):
        #     Outer stabilizers are measured via logical ancilla blocks and
        #     TRANSVERSAL CNOTs that act on ALL physical qubits in each block,
        #     not just the logical operator support. Use get_outer_x_stab_blocks()
        #     and get_transversal_outer_x_support() for this interpretation.
        #
        # Both are valid - (A) is the mathematical stabilizer, (B) is how we
        # measure it fault-tolerantly in the hierarchical scheme.
        #
        # For each outer X-check, replace each X on qubit q with inner logical X
        for row in outer.hx:
            outer_pauli: PauliString = {i: 'X' for i, bit in enumerate(row) if bit}
            lifted = lift_pauli_through_inner(
                outer_pauli, inner_log_x, inner_log_z, n_eff_inner
            )
            # Convert lifted PauliString to binary row
            # Since outer X stabilizers lift to X-type, this goes in hx
            new_row = np.zeros(n_concat, dtype=np.uint8)
            for q, op in lifted.items():
                if op in ('X', 'Y'):  # X or Y has X component
                    new_row[q] = 1
            hx_rows.append(new_row)
        
        # For each outer Z-check, replace each Z on qubit q with inner logical Z
        for row in outer.hz:
            outer_pauli: PauliString = {i: 'Z' for i, bit in enumerate(row) if bit}
            lifted = lift_pauli_through_inner(
                outer_pauli, inner_log_x, inner_log_z, n_eff_inner
            )
            # Convert to binary row for hz
            new_row = np.zeros(n_concat, dtype=np.uint8)
            for q, op in lifted.items():
                if op in ('Z', 'Y'):  # Z or Y has Z component
                    new_row[q] = 1
            hz_rows.append(new_row)

        # Stack into matrices
        hx = np.vstack(hx_rows) if hx_rows else np.zeros((0, n_concat), dtype=np.uint8)
        hz = np.vstack(hz_rows) if hz_rows else np.zeros((0, n_concat), dtype=np.uint8)

        # --- 3. Logical operators: outer logicals lifted through inner -------
        # Normalize outer logical operators to PauliString dicts
        outer_log_x_normalized = _normalize_logical_ops(outer.logical_x_ops)
        outer_log_z_normalized = _normalize_logical_ops(outer.logical_z_ops)
        
        logical_x: List[PauliString] = []
        logical_z: List[PauliString] = []
        
        for outer_log_x in outer_log_x_normalized:
            lifted = lift_pauli_through_inner(
                outer_log_x, inner_log_x, inner_log_z, n_eff_inner
            )
            logical_x.append(lifted)
            
        for outer_log_z in outer_log_z_normalized:
            lifted = lift_pauli_through_inner(
                outer_log_z, inner_log_x, inner_log_z, n_eff_inner
            )
            logical_z.append(lifted)

        # Build metadata
        meta: Dict[str, Any] = dict(metadata or {})
        meta.setdefault("outer_name", outer.name)
        meta.setdefault("inner_name", inner.name)
        meta.setdefault("n_outer", self._n_outer)
        meta.setdefault("n_inner", self._n_inner)
        meta.setdefault("depth", depth)
        meta.setdefault("effective_n_inner", n_eff_inner)
        
        # Concatenated code metadata for decoder compatibility
        meta["is_concatenated"] = True
        meta["supports_standard_decoders"] = True  # Standard decoders work via CSSMemoryExperiment
        meta["supports_hierarchical_decoder"] = True
        meta["inner_k"] = inner.k  # Number of logical qubits in inner code
        meta["outer_k"] = outer.k  # Number of logical qubits in outer code

        # Let CSSCode set up n, k, stabilizers, etc.
        CSSCode.__init__(
            self,
            hx=hx,
            hz=hz,
            logical_x=logical_x,
            logical_z=logical_z,
            metadata=meta,
        )
        
        # Store effective inner for reference
        self._effective_inner = effective_inner
        
        # Build metachecks from inner code (if available)
        # Concatenated metachecks are block-diagonal copies of inner metachecks
        self._meta_x: Optional[np.ndarray] = None
        self._meta_z: Optional[np.ndarray] = None
        self._build_concatenated_metachecks(effective_inner)

    def _build_concatenated_metachecks(self, effective_inner: CSSCode) -> None:
        """
        Build concatenated metachecks from inner code metachecks.
        
        For a concatenated code with n_outer blocks, each using an inner code
        with metachecks, the concatenated metachecks are block-diagonal:
        
            meta_concat = diag(meta_inner, meta_inner, ..., meta_inner)
                                └── n_outer copies ──┘
        
        This preserves the metacheck property: meta @ H = 0 (mod 2)
        since each block's metachecks check only that block's syndrome.
        """
        # Check if inner code has metachecks
        inner_meta_x = getattr(effective_inner, 'meta_x', None)
        inner_meta_z = getattr(effective_inner, 'meta_z', None)
        
        n_blocks = self._n_outer
        n_inner = effective_inner.n
        n_concat = self.n
        
        # Build block-diagonal meta_x (checks Z syndrome)
        if inner_meta_x is not None and inner_meta_x.size > 0:
            num_meta_rows = inner_meta_x.shape[0]
            # Concatenated meta_x acts on hz rows (Z stabilizers)
            # Inner meta_x has shape (num_meta, num_inner_z_stabs)
            # We need to map to the concatenated Z stabilizer structure
            
            # For simplicity, build meta_x that checks inner Z syndromes block-by-block
            # Total meta rows = n_blocks * num_meta_rows
            # Acts on inner Z stabilizers (first part of concatenated hz)
            num_inner_z_stabs = effective_inner.hz.shape[0]
            total_inner_z_stabs = n_blocks * num_inner_z_stabs
            
            meta_x_rows = []
            for block_idx in range(n_blocks):
                offset = block_idx * num_inner_z_stabs
                for meta_row in inner_meta_x:
                    new_row = np.zeros(total_inner_z_stabs, dtype=np.uint8)
                    new_row[offset:offset + len(meta_row)] = meta_row
                    meta_x_rows.append(new_row)
            
            if meta_x_rows:
                self._meta_x = np.vstack(meta_x_rows)
        
        # Build block-diagonal meta_z (checks X syndrome)
        if inner_meta_z is not None and inner_meta_z.size > 0:
            num_meta_rows = inner_meta_z.shape[0]
            num_inner_x_stabs = effective_inner.hx.shape[0]
            total_inner_x_stabs = n_blocks * num_inner_x_stabs
            
            meta_z_rows = []
            for block_idx in range(n_blocks):
                offset = block_idx * num_inner_x_stabs
                for meta_row in inner_meta_z:
                    new_row = np.zeros(total_inner_x_stabs, dtype=np.uint8)
                    new_row[offset:offset + len(meta_row)] = meta_row
                    meta_z_rows.append(new_row)
            
            if meta_z_rows:
                self._meta_z = np.vstack(meta_z_rows)

    @property
    def meta_x(self) -> Optional[np.ndarray]:
        """X-type metacheck matrix for concatenated code.
        
        Built from block-diagonal copies of inner code's meta_x.
        Checks Z syndrome parity within each inner block.
        """
        return self._meta_x
    
    @property
    def meta_z(self) -> Optional[np.ndarray]:
        """Z-type metacheck matrix for concatenated code.
        
        Built from block-diagonal copies of inner code's meta_z.
        Checks X syndrome parity within each inner block.
        """
        return self._meta_z

    @property
    def name(self) -> str:
        # Use the name defined in ConcatenatedCode
        return ConcatenatedCode.name.fget(self)  # type: ignore
    
    @property
    def effective_inner(self) -> CSSCode:
        """
        The effective inner code after applying depth.
        
        For depth=1, this is the same as self.inner.
        For depth>1, this is the recursively concatenated inner code.
        """
        return self._effective_inner
    
    # =========================================================================
    # Transversal Outer Stabilizer Methods (for Hierarchical Model)
    # =========================================================================
    
    def get_outer_x_stab_blocks(self, stab_idx: int) -> List[int]:
        """
        Get the block indices where outer X-stabilizer acts.
        
        For hierarchical syndrome extraction, outer X-stabilizer j acts on
        blocks where outer_hx[j, block] = 1. Transversal CNOTs are applied
        to ALL physical qubits in those blocks.
        
        Parameters
        ----------
        stab_idx : int
            Index of outer X-stabilizer (0 to r_x_out - 1).
            
        Returns
        -------
        List[int]
            Block indices where this outer X-stabilizer acts.
            
        Notes
        -----
        This differs from the lifted stabilizer (hx rows after inner stabilizers)
        which only touches qubits in the inner logical X support.
        For transversal measurement, we use entire blocks.
        """
        row = self.outer.hx[stab_idx]
        return [block_id for block_id, bit in enumerate(row) if bit]
    
    def get_outer_z_stab_blocks(self, stab_idx: int) -> List[int]:
        """
        Get the block indices where outer Z-stabilizer acts.
        
        Parameters
        ----------
        stab_idx : int
            Index of outer Z-stabilizer (0 to r_z_out - 1).
            
        Returns
        -------
        List[int]
            Block indices where this outer Z-stabilizer acts.
        """
        row = self.outer.hz[stab_idx]
        return [block_id for block_id, bit in enumerate(row) if bit]
    
    def get_transversal_outer_x_support(self, stab_idx: int) -> List[int]:
        """
        Get ALL physical qubits where outer X-stabilizer acts transversally.
        
        For hierarchical measurement via |+_L⟩ ancilla blocks, outer X-stabilizer j
        acts via transversal CNOTs on ALL physical qubits in the affected blocks,
        not just the inner logical X support.
        
        Parameters
        ----------
        stab_idx : int
            Index of outer X-stabilizer.
            
        Returns
        -------
        List[int]
            All physical qubit indices in the affected blocks.
        """
        blocks = self.get_outer_x_stab_blocks(stab_idx)
        n_inner = self._n_inner
        support = []
        for block_id in blocks:
            offset = block_id * n_inner
            support.extend(range(offset, offset + n_inner))
        return support
    
    def get_transversal_outer_z_support(self, stab_idx: int) -> List[int]:
        """
        Get ALL physical qubits where outer Z-stabilizer acts transversally.
        
        Parameters
        ----------
        stab_idx : int
            Index of outer Z-stabilizer.
            
        Returns
        -------
        List[int]
            All physical qubit indices in the affected blocks.
        """
        blocks = self.get_outer_z_stab_blocks(stab_idx)
        n_inner = self._n_inner
        support = []
        for block_id in blocks:
            offset = block_id * n_inner
            support.extend(range(offset, offset + n_inner))
        return support
    
    @property
    def outer_hx(self) -> np.ndarray:
        """
        Outer code's X parity check matrix.
        
        For hierarchical measurement, this tells which blocks each outer
        X-stabilizer acts on. Entry [j, k] = 1 means outer X-stab j acts
        on block k via transversal CNOTs.
        """
        return self.outer.hx
    
    @property
    def outer_hz(self) -> np.ndarray:
        """
        Outer code's Z parity check matrix.
        
        Entry [j, k] = 1 means outer Z-stab j acts on block k.
        """
        return self.outer.hz
    
    @property
    def inner_hx(self) -> np.ndarray:
        """Inner code's X parity check matrix."""
        return self._effective_inner.hx
    
    @property
    def inner_hz(self) -> np.ndarray:
        """Inner code's Z parity check matrix."""
        return self._effective_inner.hz


class ConcatenatedTopologicalCSSCode(ConcatenatedCSSCode, TopologicalCSSCode):
    """
    Concatenated topological CSS code.

    Adds a geometric layout by placing a scaled copy of the inner layout
    around each outer data qubit.
    
    Parameters
    ----------
    outer : TopologicalCSSCode
        Outer topological CSS code with qubit coordinates.
    inner : TopologicalCSSCode
        Inner topological CSS code with qubit coordinates.
    depth : int, optional
        Concatenation depth (default 1).
    metadata : dict, optional
        Additional metadata.
    scale : float, optional
        Scale factor for inner code layout. If None (default), an optimal
        scale is computed automatically to prevent inner block overlap.
        Typical values: ~20-30 for 4-qubit outer codes, ~3-5 for surface codes.
    margin : float, optional
        Safety margin for automatic scale computation (default 3.0).
        Larger values give more spacing between inner blocks.
    """

    def __init__(
        self,
        outer: TopologicalCSSCode,
        inner: TopologicalCSSCode,
        depth: int = 1,
        metadata: Optional[Dict[str, Any]] = None,
        scale: Optional[float] = None,
        margin: float = 3.0,
    ) -> None:
        # Build the concatenated CSS structure (Hx, Hz, etc.).
        ConcatenatedCSSCode.__init__(self, outer=outer, inner=inner, depth=depth, metadata=metadata)

        outer_coords = outer.qubit_coords()
        
        # For depth > 1, use effective inner coordinates
        if depth > 1 and hasattr(self._effective_inner, 'qubit_coords'):
            inner_coords = self._effective_inner.qubit_coords()
        else:
            inner_coords = inner.qubit_coords()

        # Handle codes without coordinates - generate default grid coords
        if outer_coords is None or len(outer_coords) == 0:
            outer_coords = [(i, 0) for i in range(outer.n)]
        if inner_coords is None or len(inner_coords) == 0:
            inner_coords = [(i, 0) for i in range(inner.n)]

        # Compute optimal scale automatically if not provided
        if scale is None:
            scale = _compute_optimal_scale(outer_coords, inner_coords, margin)
        
        # Store the computed/provided scale in metadata for reference
        self._metadata['scale'] = scale
        self._metadata['scale_auto_computed'] = (scale is None)

        # Compute inner bounding box to centre inner blocks.
        inner_arr = np.array(inner_coords, dtype=float)
        min_inner = inner_arr.min(axis=0)
        max_inner = inner_arr.max(axis=0)
        center_inner = 0.5 * (min_inner + max_inner)
        
        # Apply scale expansion for each depth level
        effective_scale = scale ** depth
        # First compute the center of outer qubit coords for proper centering
        outer_arr = np.array(outer_coords, dtype=float)
        outer_min = outer_arr.min(axis=0)
        outer_max = outer_arr.max(axis=0)
        center_outer = 0.5 * (outer_min + outer_max)

        # Build concatenated data qubit coordinates
        concat_coords: List[Coord2D] = []
        for outer_q, (ox, oy) in enumerate(outer_coords):
            for (ix, iy) in inner_coords:
                x = (ox - center_outer[0]) * effective_scale + (ix - center_inner[0])
                y = (oy - center_outer[1]) * effective_scale + (iy - center_inner[1])
                concat_coords.append((float(x), float(y)))

        self._qubit_coords = concat_coords
        
        # ====================================================================
        # Build stabilizer ancilla coordinates for stim circuit generation
        # ====================================================================
        # The concatenated stabilizers are:
        # 1. Inner stabilizers: block-diagonal, one block per outer qubit
        # 2. Lifted outer stabilizers: outer stabilizers acting through inner logicals
        
        # Get outer and inner stabilizer coordinate metadata
        outer_meta = getattr(outer, '_metadata', {}) or {}
        inner_meta = getattr(inner, '_metadata', {}) or {}
        
        outer_x_stab_coords = outer_meta.get('x_stab_coords', [])
        outer_z_stab_coords = outer_meta.get('z_stab_coords', [])
        inner_x_stab_coords = inner_meta.get('x_stab_coords', [])
        inner_z_stab_coords = inner_meta.get('z_stab_coords', [])
        
        # Build concatenated stabilizer ancilla coordinates
        concat_x_stab_coords: List[Coord2D] = []
        concat_z_stab_coords: List[Coord2D] = []

        
        # Part 1: Inner stabilizers - one copy per outer data qubit
        # These are placed within each inner block, using the SAME centering as data qubits
        # (center_inner is the center of inner data qubit coords, computed earlier)
        for outer_q, (ox, oy) in enumerate(outer_coords):
            # Inner X stabilizers for this block - center relative to inner data qubits
            for (sx, sy) in inner_x_stab_coords:
                x = (ox - center_outer[0]) * effective_scale + (sx - center_inner[0])
                y = (oy - center_outer[1]) * effective_scale + (sy - center_inner[1])
                concat_x_stab_coords.append((float(x), float(y)))
            
            # Inner Z stabilizers for this block - center relative to inner data qubits
            for (sz_x, sz_y) in inner_z_stab_coords:
                x = (ox - center_outer[0]) * effective_scale + (sz_x - center_inner[0])
                y = (oy - center_outer[1]) * effective_scale + (sz_y - center_inner[1])
                concat_z_stab_coords.append((float(x), float(y)))
        
        # Part 2: Lifted outer stabilizers - at scaled outer stabilizer positions
        # These stabilizers span across inner blocks, so we scale them similarly to outer qubits
        
        # Scale outer stabilizer positions relative to outer center
        for (sx, sy) in outer_x_stab_coords:
            x = effective_scale * (sx - center_outer[0]) 
            y = effective_scale * (sy - center_outer[1])
            concat_x_stab_coords.append((float(x), float(y)))
        
        for (sz_x, sz_y) in outer_z_stab_coords:
            x = effective_scale * (sz_x - center_outer[0]) 
            y = effective_scale * (sz_y - center_outer[1]) 
            concat_z_stab_coords.append((float(x), float(y)))
        
        # Update metadata with coordinate information for stim circuit generation
        self._metadata.update({
            'data_coords': concat_coords,
            'x_stab_coords': concat_x_stab_coords,
            'z_stab_coords': concat_z_stab_coords,
            # Preserve outer/inner info
            'outer_x_stab_count': len(outer_x_stab_coords),
            'outer_z_stab_count': len(outer_z_stab_coords),
            'inner_x_stab_count': len(inner_x_stab_coords),
            'inner_z_stab_count': len(inner_z_stab_coords),
        })
        
        # Store references for concatenated decoder support
        self._outer_code = outer
        self._inner_code = inner

    def qubit_coords(self) -> List[Coord2D]:
        """Return 2D coordinates for all physical qubits."""
        return self._qubit_coords
    
    def build_concatenation_decoder_metadata(
        self,
        rounds: int = 3,
        noise_model: Optional[Any] = None,
        basis: str = "Z",
    ) -> Dict[str, Any]:
        """
        Build metadata required for ConcatenatedDecoder.
        
        This method generates per-level DEMs and detector mappings needed for
        hierarchical decoding of concatenated codes. Uses ConcatenatedMemoryExperiment
        to ensure detector ordering matches the hierarchical structure expected by
        ConcatenatedDecoder.
        
        Parameters
        ----------
        rounds : int
            Number of syndrome measurement rounds.
        noise_model : optional
            Noise model to apply. If None, uses CircuitDepolarizingNoise(p1=0.001, p2=0.001).
        basis : str
            Measurement basis ('X' or 'Z').
            
        Returns
        -------
        Dict[str, Any]
            Dictionary with keys:
            - 'dem_per_level': List of DEMs, one per concatenation level
            - 'dem_slices': Detector index slices for each level
            - 'logicals_per_level': Number of logical qubits per level
            - 'inner_dem': DEM for a single inner code block
            - 'outer_dem': DEM for the outer code
            - 'global_dem': DEM for the full concatenated code with block-grouped detectors
        """
        from qectostim.experiments.concatenated_memory import ConcatenatedMemoryExperiment
        from qectostim.experiments.memory import CSSMemoryExperiment
        from qectostim.noise.models import CircuitDepolarizingNoise
        
        if noise_model is None:
            noise_model = CircuitDepolarizingNoise(p1=0.001, p2=0.001)
        
        # Generate the GLOBAL DEM using ConcatenatedMemoryExperiment
        # This ensures detector ordering matches hierarchical structure
        concat_exp = ConcatenatedMemoryExperiment(
            code=self,
            rounds=rounds,
            noise_model=noise_model,
            basis=basis,
        )
        concat_circuit = noise_model.apply(concat_exp.to_stim())
        try:
            global_dem = concat_circuit.detector_error_model(decompose_errors=True)
        except Exception:
            global_dem = concat_circuit.detector_error_model(
                decompose_errors=True,
                ignore_decomposition_failures=True
            )
        
        # Get detector slices from the experiment - this matches the actual circuit structure
        exp_slices = concat_exp.get_detector_slices()
        inner_slices = exp_slices['inner_slices']
        outer_slices = exp_slices['outer_slices']
        inner_dets_per_block = exp_slices['inner_dets_per_block']
        outer_dets = exp_slices['outer_dets']
        
        # Generate inner code DEM (for a single inner block)
        inner_exp = CSSMemoryExperiment(
            code=self._inner_code,
            rounds=rounds,
            noise_model=noise_model,
            basis=basis,
        )
        inner_circuit = noise_model.apply(inner_exp.to_stim())
        try:
            inner_dem = inner_circuit.detector_error_model(decompose_errors=True)
        except Exception:
            inner_dem = inner_circuit.detector_error_model(
                decompose_errors=True, 
                ignore_decomposition_failures=True
            )
        
        # Generate outer code DEM  
        outer_exp = CSSMemoryExperiment(
            code=self._outer_code,
            rounds=rounds,
            noise_model=noise_model,
            basis=basis,
        )
        outer_circuit = noise_model.apply(outer_exp.to_stim())
        try:
            outer_dem = outer_circuit.detector_error_model(decompose_errors=True)
        except Exception:
            outer_dem = outer_circuit.detector_error_model(
                decompose_errors=True,
                ignore_decomposition_failures=True
            )
        
        # For 2-level concatenation (outer ∘ inner):
        # Level 0 (inner): n_outer copies of inner DEM
        # Level 1 (outer): outer DEM
        # 
        # Use slices from the experiment, NOT from CSSMemoryExperiment detector counts,
        # because ConcatenatedMemoryExperiment has different detector structure.
        
        n_outer = self._n_outer
        
        # Note: inner_slices and outer_slices come from exp_slices above
        # inner_dets_per_block and outer_dets also come from exp_slices
        
        total_expected_detectors = n_outer * inner_dets_per_block + outer_dets
        
        concat_meta = {
            'dem_per_level': [inner_dem, outer_dem],
            'dem_slices': [inner_slices, outer_slices],
            'logicals_per_level': [self._inner_code.k, self._outer_code.k],
            'inner_dem': inner_dem,
            'outer_dem': outer_dem,
            'global_dem': global_dem,
            'n_inner_blocks': n_outer,
            'inner_n_detectors': inner_dets_per_block,  # From experiment, not CSSMemoryExperiment
            'outer_n_detectors': outer_dets,  # From experiment
            'total_expected_detectors': total_expected_detectors,
            'global_dem_n_detectors': global_dem.num_detectors,
        }
        
        # Store in metadata
        self._metadata['concatenation'] = concat_meta
        
        return concat_meta
