"""Hypergraph Product Codes

DEPRECATED: This module has been consolidated into 
qectostim.codes.composite.homological_product

This module re-exports the main classes for backward compatibility.

Hypergraph product codes are a family of QLDPC codes constructed from 
the tensor product of two chain complexes. Given CSS codes with chain
complexes A and B, the quantum code has:

  Chain length = len(A) + len(B) - 1
  Qubit grade = grade(A) + grade(B)

Examples:
  - RepetitionCode (2-chain) ⊗ RepetitionCode (2-chain) → ToricCode (3-chain)
  - ToricCode (3-chain) ⊗ ToricCode (3-chain) → 4D Tesseract (5-chain)

Reference: Tillich & Zémor, "Quantum LDPC Codes With Positive Rate and 
Minimum Distance Proportional to the Square Root of the Blocklength"
"""
import warnings

# Re-export from the consolidated module
from qectostim.codes.composite.homological_product import (
    HypergraphProductCode,
    HomologicalProductCode,
    hypergraph_product,
    homological_product,
    hypergraph_product_from_classical,
)

# Emit deprecation warning on import
warnings.warn(
    "qectostim.codes.qldpc.hypergraph_product is deprecated. "
    "Use qectostim.codes.composite.homological_product instead.",
    DeprecationWarning,
    stacklevel=2,
)


# Legacy functions that wrap the new API
def create_hgp_repetition(n: int = 3) -> HypergraphProductCode:
    """Create HGP code from repetition code.
    
    DEPRECATED: Use hypergraph_product(RepetitionCode(n), RepetitionCode(n)) instead.
    
    Uses the [n, 1, n] repetition code. Results in a quantum code
    with toric-code-like structure (3-chain).
    """
    import numpy as np
    
    # Parity check matrix for [n,1,n] repetition code
    H = np.zeros((n-1, n), dtype=np.uint8)
    for i in range(n-1):
        H[i, i] = 1
        H[i, i+1] = 1
    
    return hypergraph_product_from_classical(H, H, metadata={"base_code": f"Repetition_{n}"})


def create_hgp_hamming(m: int = 3) -> HypergraphProductCode:
    """Create HGP code from Hamming code.
    
    Uses the [2^m-1, 2^m-m-1, 3] Hamming code.
    """
    import numpy as np
    
    n = 2**m - 1
    
    # Hamming parity check matrix: columns are binary representations of 1 to n
    H = np.zeros((m, n), dtype=np.uint8)
    for col in range(n):
        val = col + 1
        for row in range(m):
            H[row, col] = (val >> row) & 1
    
    return hypergraph_product_from_classical(H, H, metadata={"base_code": f"Hamming_{m}"})


# Pre-configured instances (factory functions)
HGPHamming7 = lambda: create_hgp_hamming(m=3)  # Hamming [7,4,3] → HGP code
HGPRep5 = lambda: create_hgp_repetition(n=5)  # Repetition [5,1,5] → HGP code

__all__ = [
    "HypergraphProductCode",
    "HomologicalProductCode", 
    "hypergraph_product",
    "homological_product",
    "hypergraph_product_from_classical",
    "create_hgp_repetition",
    "create_hgp_hamming",
    "HGPHamming7",
    "HGPRep5",
]