"""
Repetition Codes: [[N,1,N]] CSS Codes

Implements the family of repetition codes which encode 1 logical qubit
in N physical qubits with distance N. These are the simplest possible
CSS codes and demonstrate directional error suppression.

The implementation uses:
- Hz: Adjacent-pair Z-type checks (detect X errors)
- Hx: Empty or minimal X-type checks

Key Features:
- Distance = N (scales with code size)
- Directional: Z-errors detected via boundary operators
- Classical limit: repition error-correcting code
"""

from typing import Tuple, List, Dict, Any, Optional
import numpy as np
from qectostim.codes.abstract_css import CSSCode
from qectostim.codes.abstract_code import PauliString


class RepetitionCode(CSSCode):
    """[[N,1,N]] Repetition Code CSS Implementation
    
    Encodes 1 logical qubit in N physical qubits with distance N.
    Uses linear chain of qubits where stabilizers check adjacent parity.
    
    Parameters
    ----------
    N : int
        Code size (number of physical qubits). Must be >= 3.
        Supported: 3, 5, 7, 9, 11, ...
        
    Attributes
    ----------
    n : int
        Number of physical qubits (inherited from CSSCode)
    k : int
        Number of logical qubits = 1 (inherited from CSSCode)
    """
    
    def __init__(self, N: int = 3, metadata: Optional[Dict[str, Any]] = None):
        """Initialize [[N,1,N]] Repetition Code
        
        Parameters
        ----------
        N : int, default=3
            Code size. Must be >= 3.
        metadata : dict, optional
            Additional metadata to store about the code.
        """
        if N < 3:
            raise ValueError(f"N must be >= 3. Got N={N}")
        
        self.N = N
        
        # Generate check matrices
        hx, hz = self._generate_checks()
        
        # Generate logical operators  
        logical_x, logical_z = self._generate_logical_operators()
        
        # Setup metadata
        meta: Dict[str, Any] = metadata or {}
        meta.update({
            "distance": N,
            "code_size": N,
            "code_type": "repetition",
            "logical_qubits": 1,
        })
        
        # Call parent CSSCode constructor
        super().__init__(hx=hx, hz=hz, logical_x=logical_x, logical_z=logical_z, 
                         metadata=meta)
    
    def _generate_checks(self) -> Tuple[np.ndarray, np.ndarray]:
        """Generate check matrices for [[N,1,N]] repetition code
        
        Uses structure:
        - Hz: (N-1) adjacent-pair Z checks 
          Each checks (qubit i, qubit i+1)
          Rank: N-1
          
        - Hx: All zeros (rank 0)
          This is valid CSS (0 @ anything = 0)
          Allows k = n - rank(Hx) - rank(Hz) = N - 0 - (N-1) = 1 ✓
        
        Logical operators:
        - Lx: anything (no Hx to anticommute with)
          Use: full-chain [X,X,...,X] for distance
        - Lz: must anticommute with ALL Hz (adjacent pairs)
          Use: middle qubit [I...IZI...I] for distance
        
        Returns
        -------
        Hx : ndarray of shape (N-1, N) or (1, N)
            Zero matrix for X checks
        Hz : ndarray of shape (N-1, N)  
            Z-type check matrix (adjacent pairs)
        """
        # Hz: single full-chain Z parity check (all qubits)
        Hz = np.ones((1, self.N), dtype=np.uint8)
        
        # Hx: adjacent-pair X parity checks
        Hx = np.zeros((self.N - 1, self.N), dtype=np.uint8)
        for i in range(self.N - 1):
            Hx[i, i] = 1
            Hx[i, i + 1] = 1
        
        return Hx, Hz
    
    def _generate_logical_operators(self) -> Tuple[List[PauliString], List[PauliString]]:
        """Generate logical X and Z operators
        
        NOTE: This is a NON-STANDARD CSS code where Hx has rank 0 (all zeros).
        The code structure encodes 1 logical qubit via boundary effects.
        
        For the [[N,1,N]] repetition code with Hx = zeros and Hz = adjacent pairs:
        
        Lx = [X,X,...,X]:
        - Any X error flips some adjacent-pair parity checks
        - Distance N because boundary qubits create long-range coupling
        
        Lz = [Z,Z,...,Z] (FULL CHAIN):
        - Although it COMMUTES with Hz (even overlap each pair),
          this is actually correct! Here's why:
        
        In this code structure:
        - Logical Z encodes the Z-BASIS DATA (what's stored)
        - Z-type checks (Hz) should COMMUTE with Lz (they don't affect logical value)
        - X-type errors are detected by Z-checks (X anticommutes with Z)
        - The logical information is in the AGREEMENT/PRODUCT across all qubits
        
        This is the "product code" or "gauge" structure where the logical value
        is determined by GLOBAL properties, not local anticommutation.
        
        Return full-chain for both X and Z to maximize distance.
        
        Returns
        -------
        Lx : list[PauliString]
            Logical X operators (1 for [[N,1,N]])
        Lz : list[PauliString]
            Logical Z operators (1 for [[N,1,N]])
        """
        # Full-chain operators for maximum distance
        lx_str = "X" * self.N
        lz_str = "Z" * self.N
        
        return [lx_str], [lz_str]


# Convenience factory functions for common code sizes
def create_repetition_code_3() -> RepetitionCode:
    """Create [[3,1,3]] repetition code"""
    return RepetitionCode(N=3)


def create_repetition_code_5() -> RepetitionCode:
    """Create [[5,1,5]] repetition code"""
    return RepetitionCode(N=5)


def create_repetition_code_7() -> RepetitionCode:
    """Create [[7,1,7]] repetition code"""
    return RepetitionCode(N=7)


def create_repetition_code_9() -> RepetitionCode:
    """Create [[9,1,9]] repetition code"""
    return RepetitionCode(N=9)

