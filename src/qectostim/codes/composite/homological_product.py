# src/qectostim/codes/composite/homological_product.py
"""
Homological Product Codes: Tensor product of chain complexes.

The homological product (or hypergraph product) of two CSS codes A and B
produces a new CSS code with parameters related to both inputs.

For two codes with chain complexes:
- Code A: n-chain complex with qubit_grade = q_A
- Code B: m-chain complex with qubit_grade = q_B

The tensor product produces:
- (n+m-1)-chain complex with qubit_grade = q_A + q_B

Examples:
- RepetitionCode (2-chain) ⊗ RepetitionCode (2-chain) → ToricCode (3-chain)
- ToricCode (3-chain) ⊗ ToricCode (3-chain) → 4D Tesseract (5-chain)

The construction at the chain complex level:
    (A ⊗ B)_k = ⊕_{i+j=k} A_i ⊗ B_j
    ∂^{A⊗B} = ∂^A ⊗ I + I ⊗ ∂^B (over GF(2))

References
----------
- Tillich & Zémor, "Quantum LDPC codes with positive rate and minimum 
  distance proportional to n^(1/2)", 2009
- Hastings, "Quantum codes from high-dimensional manifolds", 2016
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union, overload

import numpy as np

from qectostim.codes.abstract_code import PauliString
from qectostim.codes.abstract_css import (
    CSSCode,
    CSSCodeWithComplex,
    TopologicalCSSCode,
    TopologicalCSSCode3D,
    TopologicalCSSCode4D,
)
from qectostim.codes.abstract_homological import HomologicalCode
from qectostim.codes.complexes.chain_complex import ChainComplex, tensor_product_chain_complex
from qectostim.codes.complexes.css_complex import (
    CSSChainComplex2,
    CSSChainComplex3,
    CSSChainComplex4,
    FiveCSSChainComplex,
)


def _compute_logicals_from_complex(
    hx: np.ndarray,
    hz: np.ndarray,
) -> Tuple[List[PauliString], List[PauliString]]:
    """
    Compute logical operators from Hx and Hz using CSS kernel/image.
    
    Uses the CSS prescription:
    - Logical Z: ker(Hx) / rowspace(Hz)
    - Logical X: ker(Hz) / rowspace(Hx)
    """
    from qectostim.codes.utils import compute_css_logicals, vectors_to_paulis_x, vectors_to_paulis_z
    
    try:
        log_x_vecs, log_z_vecs = compute_css_logicals(hx, hz)
        logical_x = vectors_to_paulis_x(log_x_vecs)
        logical_z = vectors_to_paulis_z(log_z_vecs)
    except Exception:
        logical_x = []
        logical_z = []
    
    return logical_x, logical_z


class HomologicalProductCode(CSSCodeWithComplex):
    """
    Homological product of two CSS codes with chain complexes.
    
    For codes with 3-chain or higher complexes, computes the full tensor 
    product of chain complexes. The result has:
    
    - Chain length = len(A) + len(B) - 1  
    - Qubits on the middle grade: qubit_grade = max_grade // 2
    
    For 3-chain ⊗ 3-chain (e.g., ToricCode ⊗ ToricCode):
    - Produces a 5-chain complex (4D tesseract-like structure)
    - Qubits on grade 2 (middle of C4→C3→C2→C1→C0)
    - k = k_A × k_B (Künneth formula)
    
    Parameters
    ----------
    code_a, code_b : CSSCodeWithComplex
        CSS codes with 3-chain or higher complexes.
    metadata : dict, optional
        Additional metadata.
        
    References
    ----------
    - Bravyi & Hastings, "Homological product codes", arXiv:1311.0885
    - Audoux & Couvreur, "On tensor products of CSS codes", arXiv:1512.07081
    """
    
    def __init__(
        self,
        code_a: CSSCodeWithComplex,
        code_b: CSSCodeWithComplex,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.code_a = code_a
        self.code_b = code_b
        self.n_a = code_a.n
        self.n_b = code_b.n
        self.k_a = code_a.k
        self.k_b = code_b.k
        
        cc_a = code_a.chain_complex
        cc_b = code_b.chain_complex
        
        if cc_a is None or cc_b is None:
            raise ValueError("Both input codes must have chain complexes.")
        
        # Compute tensor product of chain complexes
        product_complex = tensor_product_chain_complex(cc_a, cc_b)
        
        # Derive Hx and Hz from the product complex
        qg = product_complex.qubit_grade
        
        if qg + 1 in product_complex.boundary_maps:
            hx = (product_complex.boundary_maps[qg + 1].T.astype(np.uint8) % 2)
        else:
            hx = np.zeros((0, product_complex.dim(qg)), dtype=np.uint8)
        
        if qg in product_complex.boundary_maps:
            hz = (product_complex.boundary_maps[qg].astype(np.uint8) % 2)
        else:
            hz = np.zeros((0, product_complex.dim(qg)), dtype=np.uint8)
        
        # Compute logical operators
        logical_x, logical_z = _compute_logicals_from_complex(hx, hz)
        
        # Build metadata
        meta: Dict[str, Any] = dict(metadata or {})
        meta["code_a_name"] = code_a.name
        meta["code_b_name"] = code_b.name
        meta["construction"] = "chain_complex_tensor"
        meta["chain_length_a"] = cc_a.max_grade + 1
        meta["chain_length_b"] = cc_b.max_grade + 1
        meta["product_chain_length"] = product_complex.max_grade + 1
        meta["qubit_grade"] = qg
        
        super().__init__(
            chain_complex=product_complex,
            logical_x=logical_x,
            logical_z=logical_z,
            metadata=meta,
        )
    
    def qubit_coords(self) -> List[Tuple[float, float]]:
        """Return 2D qubit coordinates for the product code.
        
        Uses a grid layout based on the tensor product structure:
        - Left sector: n_a × n_b qubits laid out in an n_a × n_b grid
        - Right sector: (m_a - n_a) × (m_b - n_b) qubits offset to the right
        """
        n = self.n
        # Use sqrt grid layout as fallback
        side = int(np.ceil(np.sqrt(n)))
        return [(float(i % side), float(i // side)) for i in range(n)]
    
    @property
    def name(self) -> str:
        return f"HomologicalProduct({self.code_a.name}, {self.code_b.name})"


# Backward-compatible alias (deprecated)
import warnings

def _deprecated_hgp_init(self, code_a, code_b, metadata=None):
    warnings.warn(
        "HypergraphProductCode is deprecated. Use HomologicalProductCode or "
        "homological_product() function instead.",
        DeprecationWarning,
        stacklevel=3,
    )
    HomologicalProductCode.__init__(self, code_a, code_b, metadata)

class HypergraphProductCode(HomologicalProductCode):
    """Deprecated: Use HomologicalProductCode instead."""
    def __init__(self, code_a, code_b, metadata=None):
        warnings.warn(
            "HypergraphProductCode is deprecated. Use HomologicalProductCode or "
            "homological_product() function instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(code_a, code_b, metadata)


# Factory function with overloads for proper return types

@overload
def hypergraph_product(
    code_a: CSSCodeWithComplex,
    code_b: CSSCodeWithComplex,
    metadata: Optional[Dict[str, Any]] = None,
) -> HypergraphProductCode: ...

def hypergraph_product(
    code_a: CSSCodeWithComplex,
    code_b: CSSCodeWithComplex,
    metadata: Optional[Dict[str, Any]] = None,
) -> HypergraphProductCode:
    """
    Create a hypergraph product code from two CSS codes with chain complexes.
    
    The product code's chain length is determined by the input codes:
    - 2-chain ⊗ 2-chain → 3-chain (RepetitionCode ⊗ RepetitionCode → ToricCode)
    - 3-chain ⊗ 3-chain → 5-chain (ToricCode ⊗ ToricCode → 4D Tesseract)
    
    Parameters
    ----------
    code_a, code_b : CSSCodeWithComplex
        CSS codes with chain complexes.
    metadata : dict, optional
        Additional metadata.
        
    Returns
    -------
    HypergraphProductCode
        The product code.
        
    Examples
    --------
    >>> from qectostim.codes.small import RepetitionCode
    >>> rep3 = RepetitionCode(3)
    >>> toric_like = hypergraph_product(rep3, rep3)
    >>> print(f"Chain length: {toric_like.chain_length}")  # 3
    """
    return HypergraphProductCode(code_a, code_b, metadata=metadata)


def homological_product(
    a: Union[CSSCodeWithComplex, HomologicalCode],
    b: Union[CSSCodeWithComplex, HomologicalCode],
    metadata: Optional[Dict[str, Any]] = None,
) -> Union["HomologicalProductCode", CSSCode]:
    """
    Build the homological tensor product of two codes.
    
    The homological product generalizes the hypergraph product to arbitrary
    chain complexes. For codes with chain complexes:
    
    - 2-chain ⊗ 2-chain: Uses Tillich-Zémor construction (n = n₁n₂ + r₁r₂, k = k₁k₂)
    - Higher chains: Uses full tensor product (chain length = len(A) + len(B) - 1)
    
    References
    ----------
    - Tillich & Zémor, "Quantum LDPC codes", 2009 (for 2-chain case)
    - Audoux & Couvreur, "On tensor products of CSS codes", arXiv:1512.07081
    - Bravyi & Hastings, "Homological product codes", arXiv:1311.0885
    
    Parameters
    ----------
    a, b : CSSCodeWithComplex or HomologicalCode
        The two codes to combine.
    metadata : dict, optional
        Additional metadata.
        
    Returns
    -------
    HomologicalProductCode or CSSCode
        The product code. For 2-chains, returns TillichZemorHGP.
        
    Raises
    ------
    NotImplementedError
        If inputs are not CSSCodeWithComplex.
        
    Examples
    --------
    >>> from qectostim.codes.base import RepetitionCode
    >>> rep3 = RepetitionCode(3)  # 2-chain [[3, 1, 3]]
    >>> product = homological_product(rep3, rep3)
    >>> print(f"[[{product.n}, {product.k}]]")  # [[13, 1]]
    """
    if not (isinstance(a, CSSCodeWithComplex) and isinstance(b, CSSCodeWithComplex)):
        raise NotImplementedError(
            "homological_product requires CSSCodeWithComplex inputs. "
            "Both codes must have chain complexes."
        )
    
    cc_a = a.chain_complex
    cc_b = b.chain_complex
    
    if cc_a is None or cc_b is None:
        raise ValueError("Both input codes must have chain complexes.")
    
    # For 2-chain complexes (classical codes as CSS), use Tillich-Zémor
    # A 2-chain has max_grade = 1 (only boundary_1 exists: C1 → C0)
    if cc_a.max_grade == 1 and cc_b.max_grade == 1:
        # Extract classical parity check matrices (the boundary_1 map)
        H1 = cc_a.boundary_maps[1]
        H2 = cc_b.boundary_maps[1]
        
        meta = dict(metadata or {})
        meta["code_a_name"] = a.name
        meta["code_b_name"] = b.name
        meta["construction"] = "tillich_zemor_from_2chains"
        
        return TillichZemorHGP(H1, H2, metadata=meta)
    
    # For higher-dimensional chain complexes, use full tensor product
    return HomologicalProductCode(a, b, metadata=metadata)


# Legacy support: function to create HGP from classical parity check matrices
def hypergraph_product_from_classical(
    H1: np.ndarray,
    H2: np.ndarray,
    metadata: Optional[Dict[str, Any]] = None,
) -> "TillichZemorHGP":
    """
    Create a hypergraph product code from classical parity check matrices.
    
    Uses the standard Tillich-Zémor construction which gives:
        n = n₁n₂ + r₁r₂  (where r = # checks)
        k = k₁k₂  (product of classical dimensions)
    
    Parameters
    ----------
    H1 : np.ndarray
        Parity check matrix of first classical code (r1 × n1).
    H2 : np.ndarray
        Parity check matrix of second classical code (r2 × n2).
    metadata : dict, optional
        Additional metadata.
        
    Returns
    -------
    TillichZemorHGP
        The Tillich-Zémor hypergraph product code.
    """
    return TillichZemorHGP(H1, H2, metadata=metadata)


class TillichZemorHGP(CSSCode):
    """
    Standard Tillich-Zémor hypergraph product code from classical codes.
    
    Given two classical codes with parity check matrices H1 (r1×n1) and 
    H2 (r2×n2), the hypergraph product code has:
    
    - n = n1*n2 + r1*r2 physical qubits
    - k = k1*k2 logical qubits (where ki = ni - ri)
    - d >= min(d1, d2)
    
    The stabilizer matrices are:
        Hx = [H1 ⊗ I_{n2}, I_{r1} ⊗ H2.T]
        Hz = [I_{n1} ⊗ H2, H1.T ⊗ I_{r2}]
    
    Parameters
    ----------
    H1, H2 : np.ndarray
        Classical parity check matrices.
    metadata : dict, optional
        Additional metadata.
    
    Examples
    --------
    >>> # Repetition code parity check
    >>> H = np.array([[1,1,0],[0,1,1]])  # [3,1,3] rep code
    >>> hgp = TillichZemorHGP(H, H)
    >>> print(f"[[{hgp.n}, {hgp.k}]]")  # [[13, 1]]
    """
    
    def __init__(
        self,
        H1: np.ndarray,
        H2: np.ndarray,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        H1 = np.array(H1, dtype=np.uint8) % 2
        H2 = np.array(H2, dtype=np.uint8) % 2
        
        r1, n1 = H1.shape
        r2, n2 = H2.shape
        
        I_n1 = np.eye(n1, dtype=np.uint8)
        I_n2 = np.eye(n2, dtype=np.uint8)
        I_r1 = np.eye(r1, dtype=np.uint8)
        I_r2 = np.eye(r2, dtype=np.uint8)
        
        # Tillich-Zémor construction
        # Qubits: n1*n2 (first block) + r1*r2 (second block)
        hx = np.block([[np.kron(H1, I_n2), np.kron(I_r1, H2.T)]]) % 2
        hz = np.block([[np.kron(I_n1, H2), np.kron(H1.T, I_r2)]]) % 2
        
        # Compute logicals
        logical_x, logical_z = _compute_logicals_from_complex(hx, hz)
        
        # Build metadata
        meta: Dict[str, Any] = dict(metadata or {})
        meta["construction"] = "tillich_zemor"
        meta["n1"] = n1
        meta["n2"] = n2
        meta["r1"] = r1
        meta["r2"] = r2
        meta["k1"] = n1 - r1
        meta["k2"] = n2 - r2
        
        # Store matrices for parent
        self._hx = hx
        self._hz = hz
        self._H1 = H1
        self._H2 = H2
        
        super().__init__(
            hx=hx,
            hz=hz,
            logical_x=logical_x,
            logical_z=logical_z,
            metadata=meta,
        )
    
    @property
    def name(self) -> str:
        return f"TillichZemorHGP({self._H1.shape}, {self._H2.shape})"
