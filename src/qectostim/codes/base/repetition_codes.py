"""
Repetition Codes: [[N,1,N]] CSS Codes

Implements the family of repetition codes which encode 1 logical qubit
in N physical qubits with distance N. These are the simplest possible
CSS codes and demonstrate directional error suppression.

Key insight: Classical repetition codes use ONLY one type of parity check:
adjacent-pair Z-checks. This detects bit-flip (X) errors via syndrome changes.
The logical information is encoded in the global parity across all qubits.

The implementation uses (matching Stim convention):
- Hz: (N-1) adjacent-pair Z-type checks (detect X errors via syndrome)
- Hx: Empty (no X-type checks - this code only detects X errors)

Key Features:
- Distance = N (scales with code size)
- Directional: detects X errors via local syndrome, no distance for Z errors
- Matches Stim repetition code structure
- k = 1 logical qubit with proper CSS structure
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
        
        Matches Stim's standard repetition code structure:
        - Hz: (N-1) adjacent-pair Z-type checks (detect X/bit-flip errors)
          Each checks (qubit i, qubit i+1)
          
        - Hx: Single row of all zeros (rank 0)
          No X-type checks. This code only detects X errors, not Z errors.
          For CSS code structure, Hx is minimal to satisfy orthogonality.
        
        With rank(Hx)=0 and rank(Hz)=N-1:
        k = n - rank(Hx) - rank(Hz) = N - 0 - (N-1) = 1 ✓
        
        Returns
        -------
        Hx : ndarray of shape (1, N) with all zeros
            Minimal X-type check matrix (for CSS structure)
        Hz : ndarray of shape (N-1, N)  
            Z-type check matrix (adjacent pairs)
        """
        # Hz: adjacent-pair Z-type parity checks (N-1 checks)
        # Matches Stim: checks Z*Z on adjacent qubits
        Hz = np.zeros((self.N - 1, self.N), dtype=np.uint8)
        for i in range(self.N - 1):
            Hz[i, i] = 1
            Hz[i, i + 1] = 1
        
        # Hx: Minimal matrix (one row of all zeros)
        # This represents the fact that this code only detects X errors
        # CSS orthogonality: 0 @ anything = 0 ✓
        Hx = np.zeros((1, self.N), dtype=np.uint8)
        
        return Hx, Hz
    
    def _generate_logical_operators(self) -> Tuple[List[PauliString], List[PauliString]]:
        """Generate logical X and Z operators
        
        For the [[N,1,N]] repetition code with Hx = zeros and Hz = adjacent pairs:
        
        Lx = [X,X,...,X] (full-chain X):
        - Full-chain X anticommutes with most stabilizer measurements at boundaries
        - Encodes logical information: parity of entire chain
        - Any X error on any qubit flips the parity
        - Distance N for detecting/correcting X errors
        
        Lz = [Z,I,...,I] (single-qubit Z):
        - Single qubit Z commutes with all Hz checks (even overlap with each pair)
        - Represents storable information in one qubit's Z-basis state
        - X errors detected via syndrome, not by anticommutation with Lz
        - This is correct for a detection-only (not correction) code
        
        Returns
        -------
        Lx : list[PauliString]
            Logical X operators (1 for [[N,1,N]])
        Lz : list[PauliString]
            Logical Z operators (1 for [[N,1,N]])
        """
        # Full-chain X operator
        lx_str = "X" * self.N
        
        # Single-qubit Z operator (first qubit)
        lz_str = "Z" + "I" * (self.N - 1)
        
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

