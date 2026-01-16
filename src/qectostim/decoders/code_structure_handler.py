# src/qectostim/decoders/code_structure_handler.py
"""
CodeStructureHandler - General utility for extracting code structure for hierarchical decoding.

This module provides a unified interface for extracting parity check matrices,
logical supports, and building syndrome lookup tables for any CSS code. It is
designed to support concatenated codes at multiple levels with automatic
detection of code properties.

Key Features:
- Auto-extract Hz/Hx matrices from any CSS code object
- Compute Z_L/X_L logical support sets
- Build syndrome→qubit lookup tables for lookup-based decoding
- Support for arbitrary concatenation levels
- General (not code-specific) implementation

Usage:
    >>> from qectostim.decoders.code_structure_handler import CodeStructureHandler
    >>> handler = CodeStructureHandler(concatenated_code)
    >>> inner_hz = handler.get_inner_hz()
    >>> outer_hz = handler.get_outer_hz()
    >>> inner_lookup = handler.get_inner_syndrome_lookup()
"""
from __future__ import annotations

import numpy as np
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, field

if TYPE_CHECKING:
    from qectostim.codes.composite.multilevel_concatenated import MultiLevelConcatenatedCode
    from qectostim.codes.abstract_css import CSSCode


# =============================================================================
# Default Code Structures (Fallbacks)
# =============================================================================

# Steane [[7,1,3]] code parity check matrix
STEANE_HZ = np.array([
    [0, 0, 0, 1, 1, 1, 1],
    [0, 1, 1, 0, 0, 1, 1],
    [1, 0, 1, 0, 1, 0, 1],
], dtype=np.uint8)

STEANE_HX = STEANE_HZ.copy()  # CSS code: same structure

STEANE_Z_SUPPORT = [0, 1, 2]  # Z_L = Z_0 Z_1 Z_2

# Shor [[9,1,3]] code parity check matrix
SHOR_HZ = np.array([
    [1, 1, 0, 0, 0, 0, 0, 0, 0],  # Z0Z1
    [0, 1, 1, 0, 0, 0, 0, 0, 0],  # Z1Z2
    [0, 0, 0, 1, 1, 0, 0, 0, 0],  # Z3Z4
    [0, 0, 0, 0, 1, 1, 0, 0, 0],  # Z4Z5
    [0, 0, 0, 0, 0, 0, 1, 1, 0],  # Z6Z7
    [0, 0, 0, 0, 0, 0, 0, 1, 1],  # Z7Z8
], dtype=np.uint8)

SHOR_HX = np.array([
    [1, 1, 1, 1, 1, 1, 0, 0, 0],  # X on first 6
    [0, 0, 0, 1, 1, 1, 1, 1, 1],  # X on last 6
], dtype=np.uint8)

SHOR_Z_SUPPORT = [0, 3, 6]  # Z_L = Z_0 Z_3 Z_6


@dataclass
class CodeInfo:
    """Information about a single code level."""
    n: int  # Number of physical qubits
    k: int  # Number of logical qubits
    d: int  # Distance
    hz: np.ndarray  # Z parity check matrix
    hx: np.ndarray  # X parity check matrix
    z_support: List[int]  # Qubits in Z_L support
    x_support: List[int]  # Qubits in X_L support
    syndrome_lookup_z: Dict[int, int] = field(default_factory=dict)
    syndrome_lookup_x: Dict[int, int] = field(default_factory=dict)
    
    @property
    def n_z_checks(self) -> int:
        return self.hz.shape[0] if self.hz.size > 0 else 0
    
    @property
    def n_x_checks(self) -> int:
        return self.hx.shape[0] if self.hx.size > 0 else 0


class CodeStructureHandler:
    """
    Handler for extracting and managing code structure for hierarchical decoding.
    
    This class provides a unified interface for accessing code properties
    across multiple concatenation levels. It automatically extracts parity
    check matrices, logical supports, and builds syndrome lookup tables.
    
    Parameters
    ----------
    code : MultiLevelConcatenatedCode or CSSCode
        The code to analyze. Can be a single CSS code or a multilevel
        concatenated code.
    metadata : Dict[str, Any], optional
        Additional metadata that may contain code structure hints.
    
    Attributes
    ----------
    n_levels : int
        Number of concatenation levels (1 for a simple code).
    level_info : List[CodeInfo]
        Code information for each level (outer to inner).
    """
    
    def __init__(
        self,
        code: Any,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.code = code
        self.metadata = metadata or {}
        self._level_info: List[CodeInfo] = []
        
        self._analyze_code_structure()
    
    @property
    def n_levels(self) -> int:
        """Number of concatenation levels."""
        return len(self._level_info)
    
    @property
    def level_info(self) -> List[CodeInfo]:
        """Code information for each level."""
        return self._level_info
    
    def _analyze_code_structure(self) -> None:
        """Analyze code and extract structure at each level."""
        # Check if multilevel concatenated code
        if hasattr(self.code, 'level_codes') and self.code.level_codes:
            for level_code in self.code.level_codes:
                info = self._extract_code_info(level_code)
                self._level_info.append(info)
        else:
            # Single code
            info = self._extract_code_info(self.code)
            self._level_info.append(info)
        
        # Build syndrome lookup tables for each level
        for info in self._level_info:
            info.syndrome_lookup_z = self._build_syndrome_lookup(info.hz)
            info.syndrome_lookup_x = self._build_syndrome_lookup(info.hx)
    
    def _extract_code_info(self, code: Any) -> CodeInfo:
        """Extract CodeInfo from a code object."""
        # Get n, k, d
        n = getattr(code, 'n', None)
        k = getattr(code, 'k', 1)
        d = getattr(code, 'd', 3)
        
        # Get Hz matrix
        hz = self._extract_matrix(code, ['hz', '_hz', 'parity_check_z', 'Hz'])
        
        # Get Hx matrix
        hx = self._extract_matrix(code, ['hx', '_hx', 'parity_check_x', 'Hx'])
        
        # If Hz found, infer n from it
        if hz.size > 0 and n is None:
            n = hz.shape[1]
        elif n is None:
            n = 7  # Default to Steane
        
        # If matrices not found, try default
        if hz.size == 0:
            hz = self._default_hz(n)
        if hx.size == 0:
            hx = self._default_hx(n)
        
        # Get Z logical support
        z_support = self._extract_logical_support(code, 'z', n)
        x_support = self._extract_logical_support(code, 'x', n)
        
        return CodeInfo(
            n=n,
            k=k,
            d=d,
            hz=hz,
            hx=hx,
            z_support=z_support,
            x_support=x_support,
        )
    
    def _extract_matrix(self, code: Any, attr_names: List[str]) -> np.ndarray:
        """Try to extract a matrix from code using various attribute names."""
        for attr in attr_names:
            matrix = getattr(code, attr, None)
            if matrix is not None:
                return np.atleast_2d(np.asarray(matrix, dtype=np.uint8))
        return np.zeros((0, 0), dtype=np.uint8)
    
    def _extract_logical_support(self, code: Any, basis: str, n: int) -> List[int]:
        """Extract logical support for Z or X basis."""
        # Try various method/attribute names
        if basis.lower() == 'z':
            method_names = ['logical_z_support', 'z_logical_support', 'lz']
            attr_names = ['z_support', 'logical_z', 'lz']
        else:
            method_names = ['logical_x_support', 'x_logical_support', 'lx']
            attr_names = ['x_support', 'logical_x', 'lx']
        
        # Try methods
        for method_name in method_names:
            method = getattr(code, method_name, None)
            if callable(method):
                try:
                    return list(method(0))
                except:
                    pass
        
        # Try attributes
        for attr_name in attr_names:
            attr = getattr(code, attr_name, None)
            if attr is not None:
                if isinstance(attr, (list, tuple, np.ndarray)):
                    return list(attr)
                elif isinstance(attr, dict) and 0 in attr:
                    return list(attr[0])
        
        # Default based on code size
        return self._default_logical_support(n, basis)
    
    def _default_hz(self, n: int) -> np.ndarray:
        """Get default Hz for common codes."""
        if n == 7:
            return STEANE_HZ.copy()
        elif n == 9:
            return SHOR_HZ.copy()
        else:
            # Return empty - caller must handle
            return np.zeros((0, n), dtype=np.uint8)
    
    def _default_hx(self, n: int) -> np.ndarray:
        """Get default Hx for common codes."""
        if n == 7:
            return STEANE_HX.copy()
        elif n == 9:
            return SHOR_HX.copy()
        else:
            return np.zeros((0, n), dtype=np.uint8)
    
    def _default_logical_support(self, n: int, basis: str) -> List[int]:
        """Get default logical support for common codes."""
        if n == 7:
            return STEANE_Z_SUPPORT.copy()
        elif n == 9:
            return SHOR_Z_SUPPORT.copy()
        else:
            # Generic: first few qubits
            return list(range(min(3, n)))
    
    def _build_syndrome_lookup(self, h_matrix: np.ndarray) -> Dict[int, int]:
        """
        Build syndrome → qubit error position lookup table.
        
        For each qubit q, computes syndrome of single-qubit error E_q,
        then maps syndrome_value → q.
        
        Parameters
        ----------
        h_matrix : np.ndarray
            Parity check matrix (shape: n_checks × n_qubits).
        
        Returns
        -------
        Dict[int, int]
            Mapping from syndrome integer value to error qubit position.
        """
        if h_matrix.size == 0:
            return {}
        
        n_qubits = h_matrix.shape[1]
        lookup = {}
        
        for q in range(n_qubits):
            # Single-qubit error vector
            error = np.zeros(n_qubits, dtype=np.uint8)
            error[q] = 1
            
            # Compute syndrome
            syndrome = (h_matrix @ error) % 2
            
            # Convert to integer (binary encoding)
            syn_val = int(sum(int(s) * (2 ** i) for i, s in enumerate(syndrome)))
            
            if syn_val > 0:  # Only store non-trivial syndromes
                lookup[syn_val] = q
        
        return lookup
    
    # =========================================================================
    # Public API - Level Access
    # =========================================================================
    
    def get_inner_code_info(self) -> CodeInfo:
        """Get CodeInfo for innermost level (index -1)."""
        return self._level_info[-1] if self._level_info else self._empty_code_info()
    
    def get_outer_code_info(self) -> CodeInfo:
        """Get CodeInfo for outermost level (index 0)."""
        return self._level_info[0] if self._level_info else self._empty_code_info()
    
    def get_level_info(self, level: int) -> CodeInfo:
        """Get CodeInfo for specific level (0 = outer, -1 = inner)."""
        if not self._level_info:
            return self._empty_code_info()
        if level < 0:
            level = len(self._level_info) + level
        return self._level_info[level] if 0 <= level < len(self._level_info) else self._empty_code_info()
    
    def _empty_code_info(self) -> CodeInfo:
        """Return empty CodeInfo."""
        return CodeInfo(
            n=0, k=0, d=0,
            hz=np.zeros((0, 0), dtype=np.uint8),
            hx=np.zeros((0, 0), dtype=np.uint8),
            z_support=[],
            x_support=[],
        )
    
    # =========================================================================
    # Public API - Convenience Methods
    # =========================================================================
    
    def get_inner_hz(self) -> np.ndarray:
        """Get Hz matrix for inner code."""
        return self.get_inner_code_info().hz
    
    def get_outer_hz(self) -> np.ndarray:
        """Get Hz matrix for outer code."""
        return self.get_outer_code_info().hz
    
    def get_inner_z_support(self) -> List[int]:
        """Get Z logical support for inner code."""
        return self.get_inner_code_info().z_support
    
    def get_outer_z_support(self) -> List[int]:
        """Get Z logical support for outer code."""
        return self.get_outer_code_info().z_support
    
    def get_inner_syndrome_lookup(self) -> Dict[int, int]:
        """Get syndrome→qubit lookup for inner code Z basis."""
        return self.get_inner_code_info().syndrome_lookup_z
    
    def get_outer_syndrome_lookup(self) -> Dict[int, int]:
        """Get syndrome→qubit lookup for outer code Z basis."""
        return self.get_outer_code_info().syndrome_lookup_z
    
    # =========================================================================
    # Public API - Syndrome Decoding
    # =========================================================================
    
    def decode_syndrome_lookup(
        self,
        syndrome: np.ndarray,
        lookup: Dict[int, int],
        logical_support: List[int],
    ) -> Tuple[int, int, bool]:
        """
        Decode a syndrome using lookup table.
        
        Parameters
        ----------
        syndrome : np.ndarray
            Syndrome bits.
        lookup : Dict[int, int]
            Syndrome value → error qubit mapping.
        logical_support : List[int]
            Qubits in logical operator support.
        
        Returns
        -------
        Tuple[int, int, bool]
            (error_position, correction_parity, success)
            - error_position: qubit with inferred error (-1 if no error or unknown)
            - correction_parity: 1 if error is in logical support, 0 otherwise
            - success: True if syndrome was in lookup table
        """
        # Convert syndrome to integer
        syn_val = int(sum(int(s) * (2 ** i) for i, s in enumerate(syndrome)))
        
        if syn_val == 0:
            return -1, 0, True  # No error
        
        if syn_val in lookup:
            error_pos = lookup[syn_val]
            correction_parity = 1 if error_pos in logical_support else 0
            return error_pos, correction_parity, True
        
        return -1, 0, False  # Unknown syndrome
    
    def compute_syndrome(
        self,
        data: np.ndarray,
        h_matrix: np.ndarray,
    ) -> np.ndarray:
        """Compute syndrome from data using parity check matrix."""
        if h_matrix.size == 0:
            return np.zeros(0, dtype=np.uint8)
        return (h_matrix @ data) % 2
    
    def compute_logical_parity(
        self,
        data: np.ndarray,
        support: List[int],
    ) -> int:
        """Compute parity over logical support."""
        parity = 0
        for idx in support:
            if idx < len(data):
                parity ^= int(data[idx])
        return parity
    
    # =========================================================================
    # Public API - Multi-level Decoding Support
    # =========================================================================
    
    def get_n_inner_blocks(self) -> int:
        """Get number of inner blocks (= n of outer code)."""
        if len(self._level_info) >= 2:
            return self._level_info[0].n
        return 1
    
    def get_total_data_qubits(self) -> int:
        """Get total number of data qubits in concatenated code."""
        if len(self._level_info) == 0:
            return 0
        total = 1
        for info in self._level_info:
            total *= info.n
        return total
    
    def get_block_indices(self, block_idx: int) -> Tuple[int, int]:
        """
        Get start and end indices for a given inner block.
        
        Parameters
        ----------
        block_idx : int
            Index of the inner block (0 to n_blocks - 1).
        
        Returns
        -------
        Tuple[int, int]
            (start_idx, end_idx) for slicing data array.
        """
        inner_n = self.get_inner_code_info().n
        start = block_idx * inner_n
        end = start + inner_n
        return start, end


# =============================================================================
# Convenience Functions
# =============================================================================

def extract_code_structure(
    code: Any,
    metadata: Optional[Dict[str, Any]] = None,
) -> CodeStructureHandler:
    """
    Factory function to create a CodeStructureHandler.
    
    Parameters
    ----------
    code : Any
        Code object (CSS code or multilevel concatenated code).
    metadata : Dict, optional
        Additional metadata with structure hints.
    
    Returns
    -------
    CodeStructureHandler
        Handler instance for the code.
    """
    return CodeStructureHandler(code, metadata)


def build_syndrome_lookup_from_matrix(
    h_matrix: np.ndarray,
) -> Dict[int, int]:
    """
    Build syndrome lookup table from parity check matrix.
    
    Standalone function for use without CodeStructureHandler.
    
    Parameters
    ----------
    h_matrix : np.ndarray
        Parity check matrix.
    
    Returns
    -------
    Dict[int, int]
        Syndrome value → error qubit mapping.
    """
    handler = CodeStructureHandler.__new__(CodeStructureHandler)
    return handler._build_syndrome_lookup(h_matrix)
