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

Chain Complex Structure:
- 2-chain (C1 → C0): C1 = qubits (N), C0 = Z-checks (N-1)
- boundary_1: (N-1) × N matrix mapping qubits to adjacent-pair checks
- This is the simplest chain complex, foundational for hypergraph products
"""

from typing import Tuple, List, Dict, Any, Optional
import numpy as np
from qectostim.codes.abstract_css import CSSCodeWithComplex, Coord2D
from qectostim.codes.abstract_code import PauliString
from qectostim.codes.complexes.css_complex import CSSChainComplex2


class RepetitionCode(CSSCodeWithComplex):
    """[[N,1,N]] Repetition Code with 2-Chain Complex
    
    Encodes 1 logical qubit in N physical qubits with distance N.
    Uses linear chain of qubits where stabilizers check adjacent parity.
    
    The code is defined by a 2-chain complex:
        C1 (qubits) --∂1--> C0 (checks)
    
    where ∂1 is the (N-1) × N boundary map with adjacent pairs.
    
    This is the fundamental building block for hypergraph products:
    - RepetitionCode ⊗ RepetitionCode → ToricCode (3-chain)
    - ToricCode ⊗ ToricCode → 4D Tesseract (5-chain)
    
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
    chain_complex : CSSChainComplex2
        The 2-chain complex defining the code structure
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
        
        self._N = N
        
        # Build the 2-chain complex
        chain_complex = self._build_chain_complex(N)
        
        # Generate logical operators  
        logical_x, logical_z = self._generate_logical_operators(N)
        
        # Setup metadata
        # Generate linear chain coordinates
        data_coords = [(float(i), 0.0) for i in range(N)]
        # Z stabilizer coordinates (between adjacent qubits)
        z_stab_coords = [(float(i) + 0.5, 0.0) for i in range(N - 1)]
        
        meta: Dict[str, Any] = metadata or {}
        meta.update({
            "name": f"Repetition_{N}",
            "n": N,
            "k": 1,
            "distance": N,
            "code_size": N,
            "code_type": "repetition",
            "logical_qubits": 1,
            "data_coords": data_coords,
            "x_stab_coords": [],  # No X stabilizers
            "z_stab_coords": z_stab_coords,
            "z_schedule": [(0.5, 0.0), (-0.5, 0.0)],  # Adjacent pair check
        })
        
        # Call parent CSSCodeWithComplex constructor
        super().__init__(
            chain_complex=chain_complex,
            logical_x=logical_x,
            logical_z=logical_z,
            metadata=meta,
        )
    
    @property
    def N(self) -> int:
        """Code size (number of physical qubits)."""
        return self._N
    
    @staticmethod
    def _build_chain_complex(N: int) -> CSSChainComplex2:
        """Build the 2-chain complex for the repetition code.
        
        The chain complex is:
            C1 (N qubits) --∂1--> C0 (N-1 checks)
        
        where ∂1[i, j] = 1 if qubit j is in check i.
        For adjacent-pair checks: ∂1[i, i] = ∂1[i, i+1] = 1.
        
        Parameters
        ----------
        N : int
            Number of qubits
            
        Returns
        -------
        CSSChainComplex2
            The 2-chain complex with boundary_1 = Hz
        """
        # boundary_1: (N-1) × N matrix
        # Each row i has 1s at positions i and i+1
        boundary_1 = np.zeros((N - 1, N), dtype=np.uint8)
        for i in range(N - 1):
            boundary_1[i, i] = 1
            boundary_1[i, i + 1] = 1
        
        return CSSChainComplex2(boundary_1=boundary_1)
    
    @staticmethod
    def _generate_logical_operators(N: int) -> Tuple[List[PauliString], List[PauliString]]:
        """Generate logical X and Z operators
        
        For the [[N,1,N]] repetition code:
        
        Lx = [X,X,...,X] (full-chain X):
        - Full-chain X anticommutes with most stabilizer measurements at boundaries
        - Encodes logical information: parity of entire chain
        - Distance N for detecting/correcting X errors
        
        Lz = [Z,I,...,I] (single-qubit Z):
        - Single qubit Z commutes with all Hz checks (even overlap with each pair)
        - Represents storable information in one qubit's Z-basis state
        
        Returns
        -------
        Lx : list[PauliString]
            Logical X operators (1 for [[N,1,N]])
        Lz : list[PauliString]
            Logical Z operators (1 for [[N,1,N]])
        """
        # Full-chain X operator
        lx_str = "X" * N
        
        # Single-qubit Z operator (first qubit)
        lz_str = "Z" + "I" * (N - 1)
        
        return [lx_str], [lz_str]
    
    @property
    def chain_complex(self) -> CSSChainComplex2:
        """The underlying 2-chain complex."""
        return self._chain_complex
    
    def qubit_coords(self) -> List[Coord2D]:
        """Return qubit coordinates for visualization (linear chain)."""
        return list(self._metadata.get("data_coords", []))


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

