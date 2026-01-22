# src/qectostim/codes/composite/multilevel_concatenated.py
"""
Multi-Level Concatenated Codes: Recursive Code Tree Structure.

This module provides multi-level concatenation support where codes can be 
concatenated to arbitrary depth:

    code_0 ⊗ code_1 ⊗ ... ⊗ code_{L-1}

Where code_0 is the outermost (logical) level and code_{L-1} is the innermost
(physical) level.

Key Classes
-----------
CodeNode
    Represents a node in the concatenation tree with address-based lookup.
    
MultiLevelConcatenatedCode  
    Multi-level concatenation via recursive ConcatenatedCSSCode chaining.
    Provides both tree-based hierarchical access and flat matrix access.

ConcatenatedCodeBuilder
    Fluent builder API for constructing multi-level codes.

Example
-------
>>> from qectostim.codes.small import SteaneCode713, ShorCode91
>>> from qectostim.codes.small.hamming_css import HammingCSS15
>>> 
>>> # Build 3-level concatenation: Hamming[[15,1,3]] ⊗ Steane[[7,1,3]] ⊗ Shor[[9,1,3]]
>>> code = (ConcatenatedCodeBuilder()
...     .add_level(HammingCSS15())  # outermost
...     .add_level(SteaneCode713())
...     .add_level(ShorCode91())    # innermost
...     .build())
>>> 
>>> code.n  # 15 * 7 * 9 = 945 physical qubits
945
>>> code.depth
3
>>> code.total_distance  # 3 * 3 * 3 = 27
27
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, Iterator, List, Optional, Tuple

import numpy as np

from qectostim.codes.composite.concatenated import ConcatenatedCSSCode

if TYPE_CHECKING:
    from qectostim.codes.abstract_css import CSSCode


@dataclass
class CodeNode:
    """
    Node in the concatenation tree.
    
    Each node represents a block at a particular level of concatenation.
    Leaf nodes are at the innermost (physical) level.
    
    Attributes
    ----------
    code : CSSCode
        The CSS code at this level (same for all nodes at same level).
    level : int
        Level in the tree (0 = outermost/root, depth-1 = innermost/leaves).
    children : List[CodeNode]
        Child nodes (empty for leaf nodes).
    address : Tuple[int, ...]
        Hierarchical address from root (empty for root, (i,) for root's i-th child, etc.).
        
    Properties
    ----------
    is_leaf : bool
        True if this node has no children (innermost level).
    n_physical_total : int
        Total physical qubits under this node.
    """
    code: Any  # CSSCode
    level: int
    children: List['CodeNode'] = field(default_factory=list)
    address: Tuple[int, ...] = ()
    
    @property
    def is_leaf(self) -> bool:
        """True if this is a leaf node (no children)."""
        return len(self.children) == 0
    
    @property
    def n_physical_total(self) -> int:
        """Total physical qubits under this node."""
        if self.is_leaf:
            return self.code.n
        return sum(c.n_physical_total for c in self.children)
    
    @property
    def n_blocks(self) -> int:
        """Number of immediate child blocks (= code.n for non-leaf)."""
        return len(self.children) if self.children else 0
    
    def __repr__(self) -> str:
        code_name = getattr(self.code, 'name', type(self.code).__name__)
        return f"CodeNode(level={self.level}, address={self.address}, code={code_name})"


class MultiLevelConcatenatedCode:
    """
    Multi-level concatenation via chained ConcatenatedCSSCode.
    
    Provides two complementary views:
    1. **Tree view**: Hierarchical CodeNode tree with address-based lookup.
    2. **Matrix view**: Flat concatenated Hz/Hx matrices via chained ConcatenatedCSSCode.
    
    The concatenation order is:
        codes[0] ⊗ codes[1] ⊗ ... ⊗ codes[-1]
    
    Where codes[0] is the outermost (fewest physical qubits per logical) and
    codes[-1] is the innermost (physical layer).
    
    Parameters
    ----------
    codes : List[CSSCode]
        List of codes from outermost to innermost.
        
    Attributes
    ----------
    level_codes : List[CSSCode]
        The codes at each level.
    depth : int
        Number of concatenation levels.
    root : CodeNode
        Root of the code tree.
    concatenated : ConcatenatedCSSCode
        Fully concatenated code for matrix access.
        
    Example
    -------
    >>> from qectostim.codes.small import SteaneCode713, ShorCode91
    >>> 
    >>> code = MultiLevelConcatenatedCode([SteaneCode713(), ShorCode91()])
    >>> code.n  # 7 * 9 = 63
    63
    >>> code.depth
    2
    >>> list(code.iter_leaves())  # 7 leaf nodes
    [CodeNode(level=1, address=(0,), ...), ...]
    """
    
    def __init__(self, codes: List[Any]) -> None:
        """
        Build multi-level concatenation.
        
        Parameters
        ----------
        codes : List[CSSCode]
            List of codes from outermost (index 0) to innermost (index -1).
            Must have at least 2 codes.
        """
        if len(codes) < 2:
            raise ValueError(f"Need at least 2 codes for concatenation, got {len(codes)}")
        
        self.level_codes = list(codes)
        self.depth = len(codes)
        
        # Build the code tree (for hierarchical access)
        self._build_tree()
        
        # Build the chained ConcatenatedCSSCode (for matrix access)
        self._build_concatenated_chain()
        
        # Precompute total distance
        self._total_distance = self._compute_total_distance()
    
    def _build_tree(self) -> None:
        """Build the CodeNode tree structure."""
        self.root = self._build_subtree(0, ())
    
    def _build_subtree(self, level: int, address: Tuple[int, ...]) -> CodeNode:
        """
        Recursively build subtree starting at given level and address.
        
        Parameters
        ----------
        level : int
            Current level (0 = outermost).
        address : Tuple[int, ...]
            Address of this node.
            
        Returns
        -------
        CodeNode
            The node and its descendants.
        """
        code = self.level_codes[level]
        node = CodeNode(code=code, level=level, address=address)
        
        if level < self.depth - 1:
            # Each qubit of this code becomes an encoded block at the next level
            for i in range(code.n):
                child = self._build_subtree(level + 1, address + (i,))
                node.children.append(child)
        
        return node
    
    def _build_concatenated_chain(self) -> None:
        """
        Build chain of ConcatenatedCSSCode for matrix access.
        
        Chains from innermost outward:
            codes[-1] → (codes[-2] ⊗ codes[-1]) → (codes[-3] ⊗ previous) → ...
        """
        # Start with innermost code
        current = self.level_codes[-1]
        
        # Chain outward
        for i in range(self.depth - 2, -1, -1):
            current = ConcatenatedCSSCode(
                outer=self.level_codes[i],
                inner=current
            )
        
        self.concatenated = current
    
    def _compute_total_distance(self) -> int:
        """
        Compute total concatenated distance.
        
        For k=1 logical qubit at each level, distance multiplies:
            d_total = d_0 * d_1 * ... * d_{L-1}
        """
        total_d = 1
        for code in self.level_codes:
            d = getattr(code, 'd', getattr(code, 'distance', 3))
            total_d *= d
        return total_d
    
    # =========================================================================
    # Matrix access (delegated to concatenated chain)
    # =========================================================================
    
    @property
    def hx(self) -> np.ndarray:
        """X-type parity check matrix (detects Z errors)."""
        return self.concatenated.hx
    
    @property
    def hz(self) -> np.ndarray:
        """Z-type parity check matrix (detects X errors)."""
        return self.concatenated.hz
    
    @property
    def n(self) -> int:
        """Total number of physical qubits."""
        return self.concatenated.n
    
    @property
    def k(self) -> int:
        """Number of logical qubits."""
        return self.concatenated.k
    
    @property
    def d(self) -> int:
        """Code distance."""
        return self._total_distance
    
    @property
    def distance(self) -> int:
        """Code distance (alias for d)."""
        return self._total_distance
    
    @property
    def total_distance(self) -> int:
        """Total concatenated distance."""
        return self._total_distance
    
    @property
    def logical_x_ops(self) -> List[Any]:
        """Logical X operators."""
        return self.concatenated.logical_x_ops
    
    @property
    def logical_z_ops(self) -> List[Any]:
        """Logical Z operators."""
        return self.concatenated.logical_z_ops
    
    @property
    def name(self) -> str:
        """Human-readable name."""
        names = [getattr(c, 'name', type(c).__name__) for c in self.level_codes]
        return "MultiLevel(" + " ⊗ ".join(names) + ")"
    
    # =========================================================================
    # Tree navigation
    # =========================================================================
    
    def get_node(self, address: Tuple[int, ...]) -> CodeNode:
        """
        Get node by hierarchical address.
        
        Parameters
        ----------
        address : Tuple[int, ...]
            Hierarchical address. Empty tuple returns root.
            
        Returns
        -------
        CodeNode
            The node at the given address.
            
        Raises
        ------
        IndexError
            If address is invalid.
            
        Example
        -------
        >>> code = MultiLevelConcatenatedCode([outer, inner])
        >>> code.get_node(())      # root
        >>> code.get_node((0,))    # first child of root
        >>> code.get_node((0, 3))  # 4th grandchild of first child
        """
        node = self.root
        for idx in address:
            if idx >= len(node.children):
                raise IndexError(f"Invalid address {address}: index {idx} >= {len(node.children)}")
            node = node.children[idx]
        return node
    
    def iter_leaves(self) -> Iterator[CodeNode]:
        """
        Iterate over all leaf (physical-level) nodes.
        
        Yields
        ------
        CodeNode
            Each leaf node in depth-first order.
        """
        def recurse(node: CodeNode) -> Iterator[CodeNode]:
            if node.is_leaf:
                yield node
            else:
                for child in node.children:
                    yield from recurse(child)
        
        yield from recurse(self.root)
    
    def iter_level(self, level: int) -> Iterator[CodeNode]:
        """
        Iterate over all nodes at a given level.
        
        Parameters
        ----------
        level : int
            Level to iterate (0 = root level).
            
        Yields
        ------
        CodeNode
            Each node at the specified level.
        """
        def recurse(node: CodeNode) -> Iterator[CodeNode]:
            if node.level == level:
                yield node
            else:
                for child in node.children:
                    yield from recurse(child)
        
        yield from recurse(self.root)
    
    def count_leaves(self) -> int:
        """Count total number of leaf nodes."""
        return sum(1 for _ in self.iter_leaves())
    
    def count_nodes_at_level(self, level: int) -> int:
        """Count nodes at a given level."""
        return sum(1 for _ in self.iter_level(level))
    
    # =========================================================================
    # Level-specific access
    # =========================================================================
    
    def get_level_code(self, level: int) -> Any:
        """
        Get the code used at a specific level.
        
        Parameters
        ----------
        level : int
            Level index (0 = outermost, depth-1 = innermost).
            
        Returns
        -------
        CSSCode
            The code at that level.
        """
        return self.level_codes[level]
    
    def get_level_n(self, level: int) -> int:
        """Get number of qubits in code at given level."""
        return self.level_codes[level].n
    
    def get_level_distance(self, level: int) -> int:
        """Get distance of code at given level."""
        code = self.level_codes[level]
        return getattr(code, 'd', getattr(code, 'distance', 3))
    
    # =========================================================================
    # Block size calculations
    # =========================================================================
    
    def block_size_at_level(self, level: int) -> int:
        """
        Get the physical block size for blocks at given level.
        
        For a node at level L, its block size is the product of n values
        for all codes from level L+1 to the innermost level.
        
        Parameters
        ----------
        level : int
            Level index.
            
        Returns
        -------
        int
            Physical qubits per block at this level.
        """
        if level >= self.depth - 1:
            return self.level_codes[-1].n
        
        size = 1
        for i in range(level + 1, self.depth):
            size *= self.level_codes[i].n
        return size
    
    def total_blocks_at_level(self, level: int) -> int:
        """
        Get total number of blocks at given level.
        
        This is the product of n values for all codes from level 0 to level L-1.
        
        Parameters
        ----------
        level : int
            Level index.
            
        Returns
        -------
        int
            Total number of blocks at this level.
        """
        if level == 0:
            return 1  # Just the root
        
        count = 1
        for i in range(level):
            count *= self.level_codes[i].n
        return count
    
    # =========================================================================
    # Convenience properties for 2-level compatibility
    # =========================================================================
    
    @property
    def outer(self) -> Any:
        """Outermost code (for 2-level compatibility)."""
        return self.level_codes[0]
    
    @property
    def inner(self) -> Any:
        """Innermost code (for 2-level compatibility)."""
        return self.level_codes[-1]
    
    @property
    def n_outer(self) -> int:
        """Number of qubits in outermost code."""
        return self.level_codes[0].n
    
    @property
    def n_inner(self) -> int:
        """Number of qubits in innermost code."""
        return self.level_codes[-1].n
    
    @property
    def physical_code(self) -> Any:
        """Innermost code (decoder convention)."""
        return self.level_codes[-1]
    
    @property
    def logical_code(self) -> Any:
        """Outermost code (decoder convention)."""
        return self.level_codes[0]
    
    # =========================================================================
    # Metadata for decoder compatibility
    # =========================================================================
    
    @property
    def is_concatenated(self) -> bool:
        """True - this is a concatenated code."""
        return True
    
    @property
    def is_multilevel(self) -> bool:
        """True - this is a multi-level concatenated code."""
        return True
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Export code structure as dictionary.
        
        Useful for serialization and decoder metadata.
        """
        return {
            'depth': self.depth,
            'n': self.n,
            'k': self.k,
            'd': self.d,
            'level_codes': [
                {
                    'level': i,
                    'name': getattr(c, 'name', type(c).__name__),
                    'n': c.n,
                    'k': getattr(c, 'k', 1),
                    'd': getattr(c, 'd', getattr(c, 'distance', 3)),
                }
                for i, c in enumerate(self.level_codes)
            ],
            'total_leaves': self.count_leaves(),
        }
    
    def __repr__(self) -> str:
        return f"MultiLevelConcatenatedCode(depth={self.depth}, n={self.n}, d={self.d})"


class ConcatenatedCodeBuilder:
    """
    Fluent builder for multi-level concatenated codes.
    
    Example
    -------
    >>> code = (ConcatenatedCodeBuilder()
    ...     .add_level(HammingCSS15())   # outermost (level 0)
    ...     .add_level(SteaneCode713())   # middle (level 1)
    ...     .add_level(ShorCode91())     # innermost (level 2)
    ...     .build())
    """
    
    def __init__(self) -> None:
        """Initialize empty builder."""
        self._levels: List[Any] = []
    
    def add_level(self, code: Any) -> 'ConcatenatedCodeBuilder':
        """
        Add a code level.
        
        The first code added is the outermost; the last is the innermost.
        
        Parameters
        ----------
        code : CSSCode
            CSS code to add as a concatenation level.
            
        Returns
        -------
        ConcatenatedCodeBuilder
            Self for method chaining.
        """
        self._levels.append(code)
        return self
    
    def add_levels(self, codes: List[Any]) -> 'ConcatenatedCodeBuilder':
        """
        Add multiple code levels at once.
        
        Parameters
        ----------
        codes : List[CSSCode]
            Codes to add (first = outermost, last = innermost).
            
        Returns
        -------
        ConcatenatedCodeBuilder
            Self for method chaining.
        """
        self._levels.extend(codes)
        return self
    
    def repeat(self, code: Any, times: int) -> 'ConcatenatedCodeBuilder':
        """
        Add the same code multiple times (tower construction).
        
        Parameters
        ----------
        code : CSSCode
            Code to repeat.
        times : int
            Number of times to repeat.
            
        Returns
        -------
        ConcatenatedCodeBuilder
            Self for method chaining.
            
        Example
        -------
        >>> # Build Steane^3 (3-level Steane tower)
        >>> code = ConcatenatedCodeBuilder().repeat(SteaneCode713(), 3).build()
        """
        for _ in range(times):
            self._levels.append(code)
        return self
    
    def build(self) -> MultiLevelConcatenatedCode:
        """
        Build the multi-level concatenated code.
        
        Returns
        -------
        MultiLevelConcatenatedCode
            The constructed code.
            
        Raises
        ------
        ValueError
            If fewer than 2 levels have been added.
        """
        if len(self._levels) < 2:
            raise ValueError(
                f"Need at least 2 levels for concatenation, got {len(self._levels)}. "
                "Use add_level() to add codes."
            )
        return MultiLevelConcatenatedCode(self._levels)
    
    def __len__(self) -> int:
        """Number of levels added so far."""
        return len(self._levels)
    
    def __repr__(self) -> str:
        return f"ConcatenatedCodeBuilder(levels={len(self._levels)})"


# =============================================================================
# Convenience factory functions
# =============================================================================

def build_steane_tower(depth: int) -> MultiLevelConcatenatedCode:
    """
    Build a Steane code tower (Steane^depth).
    
    Parameters
    ----------
    depth : int
        Number of concatenation levels (minimum 2).
        
    Returns
    -------
    MultiLevelConcatenatedCode
        Steane code concatenated with itself `depth` times.
        
    Example
    -------
    >>> code = build_steane_tower(3)
    >>> code.n  # 7^3 = 343
    343
    """
    from qectostim.codes.small import SteaneCode713
    return ConcatenatedCodeBuilder().repeat(SteaneCode713(), depth).build()


def build_standard_concatenation(
    outer_name: str,
    inner_name: str,
) -> MultiLevelConcatenatedCode:
    """
    Build standard 2-level concatenation from code names.
    
    Parameters
    ----------
    outer_name : str
        Name of outer code: 'steane', 'shor', 'hamming7', 'hamming15', 'rep3', 'rep5'.
    inner_name : str
        Name of inner code (same options).
        
    Returns
    -------
    MultiLevelConcatenatedCode
        The concatenated code.
    """
    from qectostim.codes.small import (
        SteaneCode713,
        ShorCode91,
        RepetitionCode,
    )
    from qectostim.codes.small.hamming_css import HammingCSS7, HammingCSS15
    
    code_map = {
        'steane': SteaneCode713,
        'shor': ShorCode91,
        'hamming7': HammingCSS7,
        'hamming15': HammingCSS15,
        'rep3': lambda: RepetitionCode(3),
        'rep5': lambda: RepetitionCode(5),
    }
    
    outer_cls = code_map.get(outer_name.lower())
    inner_cls = code_map.get(inner_name.lower())
    
    if outer_cls is None:
        raise ValueError(f"Unknown outer code: {outer_name}")
    if inner_cls is None:
        raise ValueError(f"Unknown inner code: {inner_name}")
    
    return ConcatenatedCodeBuilder().add_level(outer_cls()).add_level(inner_cls()).build()
