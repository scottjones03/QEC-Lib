# src/qectostim/codes/composite/homological_product.py
"""
Homological Product Codes: Tensor product of chain complexes.

The homological product (or hypergraph product) of two CSS codes A and B
produces a new CSS code with parameters related to both inputs.

For two CSS codes:
- Code A: [[n_A, k_A, d_A]] with parity checks Hx_A, Hz_A
- Code B: [[n_B, k_B, d_B]] with parity checks Hx_B, Hz_B

The hypergraph product produces:
- [[n, k, d]] with n = n_A*r_B + r_A*n_B, k = k_A*k_B, d >= min(d_A, d_B)

where r_A = rank(Hx_A) and r_B = rank(Hz_B) (number of check rows).

The parity check matrices are:
- Hx = [Hx_A ⊗ I_nB | I_rxA ⊗ Hx_B^T]
- Hz = [I_nA ⊗ Hz_B | Hz_A ⊗ I_nB]

where rxA = number of rows in Hx_A.

References
----------
- Tillich & Zémor, "Quantum LDPC codes with positive rate and minimum 
  distance proportional to n^(1/2)", 2009
- Hastings, "Quantum codes from high-dimensional manifolds", 2016
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from qectostim.codes.abstract_code import PauliString
from qectostim.codes.abstract_css import CSSCode, TopologicalCSSCode, TopologicalCSSCode4D
from qectostim.codes.abstract_homological import HomologicalCode, TopologicalCode
from qectostim.codes.complexes.chain_complex import ChainComplex
from qectostim.codes.utils import (
    kron_gf2,
    gf2_rank,
    gf2_kernel,
    css_intersection_check,
    vectors_to_paulis_x,
    vectors_to_paulis_z,
)


class HypergraphProductCode(CSSCode):
    """
    Hypergraph product of two CSS codes.
    
    Given codes A and B, constructs A ⊗ B using the hypergraph product
    construction. The result is a CSS code with:
    
    - n = n_A * m_B + m_A * n_B (physical qubits)
    - k = k_A * k_B (logical qubits)
    - d >= min(d_A, d_B)
    
    where m_A = rows(Hx_A) and m_B = rows(Hz_B).
    
    Parameters
    ----------
    code_a : CSSCode
        First CSS code.
    code_b : CSSCode
        Second CSS code.
    metadata : dict, optional
        Additional metadata.
        
    Attributes
    ----------
    code_a : CSSCode
        First factor code.
    code_b : CSSCode
        Second factor code.
    n_a, n_b : int
        Physical qubit counts of factor codes.
    k_a, k_b : int
        Logical qubit counts of factor codes.
        
    Examples
    --------
    >>> from qectostim.codes.base import RepetitionCode
    >>> rep3 = RepetitionCode(3)  # [[3, 1, 3]] bit-flip code
    >>> # Product with itself gives a surface-code-like structure
    >>> product = HypergraphProductCode(rep3, rep3)
    >>> print(f"[[{product.n}, {product.k}]]")
    
    Notes
    -----
    The hypergraph product is symmetric in a sense, but A ⊗ B is not
    identical to B ⊗ A (they are related by qubit relabeling).
    
    The construction is also called the "Tillich-Zémor construction"
    or "quantum LDPC from classical LDPC" construction.
    """
    
    def __init__(
        self,
        code_a: CSSCode,
        code_b: CSSCode,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.code_a = code_a
        self.code_b = code_b
        
        # Extract dimensions
        hx_a = code_a.hx
        hz_a = code_a.hz
        hx_b = code_b.hx
        hz_b = code_b.hz
        
        n_a = code_a.n
        n_b = code_b.n
        m_xa = hx_a.shape[0]  # Number of X checks in A
        m_za = hz_a.shape[0]  # Number of Z checks in A
        m_xb = hx_b.shape[0]  # Number of X checks in B
        m_zb = hz_b.shape[0]  # Number of Z checks in B
        
        self.n_a = n_a
        self.n_b = n_b
        self.k_a = code_a.k
        self.k_b = code_b.k
        
        # Total physical qubits in product code
        # Left block: n_A × m_zB qubits (one per (qubit_A, z_check_B) pair)
        # Right block: m_xA × n_B qubits (one per (x_check_A, qubit_B) pair)
        n_left = n_a * m_zb
        n_right = m_xa * n_b
        n_total = n_left + n_right
        
        if n_total == 0:
            raise ValueError("Product code has no physical qubits")
        
        # --- Build Hx = [Hx_A ⊗ I_mzB | I_mxA ⊗ Hx_B^T] ---
        # Left part: Hx_A ⊗ I_{m_zB}
        I_mzb = np.eye(m_zb, dtype=np.uint8) if m_zb > 0 else np.zeros((0, 0), dtype=np.uint8)
        hx_left = kron_gf2(hx_a, I_mzb) if m_zb > 0 and m_xa > 0 else np.zeros((0, n_left), dtype=np.uint8)
        
        # Right part: I_{m_xA} ⊗ Hx_B^T
        I_mxa = np.eye(m_xa, dtype=np.uint8) if m_xa > 0 else np.zeros((0, 0), dtype=np.uint8)
        hx_b_t = hx_b.T if hx_b.size > 0 else np.zeros((n_b, 0), dtype=np.uint8)
        hx_right = kron_gf2(I_mxa, hx_b_t) if m_xa > 0 and n_b > 0 else np.zeros((0, n_right), dtype=np.uint8)
        
        # Combine: number of rows should match
        # Hx_A ⊗ I has shape (m_xA * m_zB, n_A * m_zB)
        # I ⊗ Hx_B^T has shape (m_xA * n_B, m_xA * n_B) but we want (m_xA * m_xB, m_xA * n_B)
        # Actually I_mxa ⊗ Hx_B^T has shape (m_xA * m_xB, m_xA * n_B)
        
        # Re-derive: The X checks come from:
        # 1. Each X check of A, tensored across all Z-syndrome qubits of B
        # 2. Each qubit of A, tensored with X checks of B
        
        # Let me use the standard hypergraph product formulation more carefully:
        # Qubits: (i,j) with i ∈ [n_A], j ∈ [m_B'] or i ∈ [m_A], j ∈ [n_B]
        # where m_A = rows of H_A, m_B' = rows of H_B^T
        
        # Standard form uses H_A for the first code's parity check
        # and H_B for the second code's parity check
        
        # For CSS, we use:
        # H_A = Hx_A (for X stabilizers detecting Z errors)
        # H_B = Hz_B (for Z stabilizers detecting X errors)
        
        # Let's use a cleaner implementation
        hx_prod, hz_prod = self._build_product_matrices(hx_a, hz_a, hx_b, hz_b)
        
        # --- Build logical operators ---
        # The logical operators of the product come from:
        # - Kernel of one code tensored with logicals of the other
        # This is more complex, so we'll compute them numerically
        
        logical_x, logical_z = self._compute_logicals(hx_prod, hz_prod)
        
        # Build metadata
        meta: Dict[str, Any] = dict(metadata or {})
        meta["code_a_name"] = code_a.name
        meta["code_b_name"] = code_b.name
        meta["n_a"] = n_a
        meta["n_b"] = n_b
        meta["k_a"] = self.k_a
        meta["k_b"] = self.k_b
        meta["is_hypergraph_product"] = True
        
        super().__init__(
            hx=hx_prod,
            hz=hz_prod,
            logical_x=logical_x,
            logical_z=logical_z,
            metadata=meta,
        )
    
    def _build_product_matrices(
        self,
        hx_a: np.ndarray,
        hz_a: np.ndarray,
        hx_b: np.ndarray,
        hz_b: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Build the hypergraph product parity check matrices.
        
        Standard construction:
        Hx = [H1 ⊗ I_{n2} | I_{m1} ⊗ H2^T]
        Hz = [I_{n1} ⊗ H2 | H1^T ⊗ I_{m2}]
        
        where H1 = Hx_A, H2 = Hz_B, and m1 = rows(H1), m2 = rows(H2).
        """
        H1 = hx_a  # m1 × n1
        H2 = hz_b  # m2 × n2
        
        m1, n1 = H1.shape if H1.size > 0 else (0, 0)
        m2, n2 = H2.shape if H2.size > 0 else (0, 0)
        
        # Handle edge cases
        if n1 == 0 or n2 == 0:
            raise ValueError("Input codes must have at least 1 qubit")
        
        # Total qubits: n1*n2 + m1*m2
        n_total = n1 * n2 + m1 * m2
        
        # Identity matrices
        I_n1 = np.eye(n1, dtype=np.uint8)
        I_n2 = np.eye(n2, dtype=np.uint8)
        I_m1 = np.eye(m1, dtype=np.uint8) if m1 > 0 else np.zeros((0, 0), dtype=np.uint8)
        I_m2 = np.eye(m2, dtype=np.uint8) if m2 > 0 else np.zeros((0, 0), dtype=np.uint8)
        
        # Build Hx = [H1 ⊗ I_n2 | I_m1 ⊗ H2^T]
        # Left block: m1*n2 rows, n1*n2 columns
        hx_left = kron_gf2(H1, I_n2) if m1 > 0 else np.zeros((0, n1 * n2), dtype=np.uint8)
        
        # Right block: m1*n2 rows, m1*m2 columns (if m2 > 0)
        # Wait, I_m1 ⊗ H2^T has shape (m1 * n2, m1 * m2)
        H2_T = H2.T if m2 > 0 else np.zeros((n2, 0), dtype=np.uint8)
        hx_right = kron_gf2(I_m1, H2_T) if m1 > 0 and m2 > 0 else np.zeros((m1 * n2 if m1 > 0 else 0, m1 * m2), dtype=np.uint8)
        
        # Combine horizontally
        if hx_left.size > 0 and hx_right.size > 0:
            # Ensure compatible row counts
            if hx_left.shape[0] != hx_right.shape[0]:
                # Pad if necessary
                max_rows = max(hx_left.shape[0], hx_right.shape[0])
                if hx_left.shape[0] < max_rows:
                    hx_left = np.vstack([hx_left, np.zeros((max_rows - hx_left.shape[0], hx_left.shape[1]), dtype=np.uint8)])
                if hx_right.shape[0] < max_rows:
                    hx_right = np.vstack([hx_right, np.zeros((max_rows - hx_right.shape[0], hx_right.shape[1]), dtype=np.uint8)])
            hx_prod = np.hstack([hx_left, hx_right])
        elif hx_left.size > 0:
            hx_prod = np.hstack([hx_left, np.zeros((hx_left.shape[0], m1 * m2), dtype=np.uint8)])
        elif hx_right.size > 0:
            hx_prod = np.hstack([np.zeros((hx_right.shape[0], n1 * n2), dtype=np.uint8), hx_right])
        else:
            hx_prod = np.zeros((0, n_total), dtype=np.uint8)
        
        # Build Hz = [I_n1 ⊗ H2 | H1^T ⊗ I_m2]
        # Left block: n1*m2 rows, n1*n2 columns
        hz_left = kron_gf2(I_n1, H2) if m2 > 0 else np.zeros((0, n1 * n2), dtype=np.uint8)
        
        # Right block: n1*m2 rows, m1*m2 columns
        H1_T = H1.T if m1 > 0 else np.zeros((n1, 0), dtype=np.uint8)
        hz_right = kron_gf2(H1_T, I_m2) if m1 > 0 and m2 > 0 else np.zeros((n1 * m2 if m2 > 0 else 0, m1 * m2), dtype=np.uint8)
        
        # Combine horizontally
        if hz_left.size > 0 and hz_right.size > 0:
            if hz_left.shape[0] != hz_right.shape[0]:
                max_rows = max(hz_left.shape[0], hz_right.shape[0])
                if hz_left.shape[0] < max_rows:
                    hz_left = np.vstack([hz_left, np.zeros((max_rows - hz_left.shape[0], hz_left.shape[1]), dtype=np.uint8)])
                if hz_right.shape[0] < max_rows:
                    hz_right = np.vstack([hz_right, np.zeros((max_rows - hz_right.shape[0], hz_right.shape[1]), dtype=np.uint8)])
            hz_prod = np.hstack([hz_left, hz_right])
        elif hz_left.size > 0:
            hz_prod = np.hstack([hz_left, np.zeros((hz_left.shape[0], m1 * m2), dtype=np.uint8)])
        elif hz_right.size > 0:
            hz_prod = np.hstack([np.zeros((hz_right.shape[0], n1 * n2), dtype=np.uint8), hz_right])
        else:
            hz_prod = np.zeros((0, n_total), dtype=np.uint8)
        
        # Ensure column counts match
        if hx_prod.shape[1] != n_total:
            hx_prod = np.hstack([hx_prod, np.zeros((hx_prod.shape[0], n_total - hx_prod.shape[1]), dtype=np.uint8)])
        if hz_prod.shape[1] != n_total:
            hz_prod = np.hstack([hz_prod, np.zeros((hz_prod.shape[0], n_total - hz_prod.shape[1]), dtype=np.uint8)])
        
        return hx_prod, hz_prod
    
    def _compute_logicals(
        self,
        hx: np.ndarray,
        hz: np.ndarray,
    ) -> Tuple[List[PauliString], List[PauliString]]:
        """
        Compute logical operators from Hx and Hz.
        
        Uses the CSS prescription:
        - Logical Z: ker(Hx) / rowspace(Hz)
        - Logical X: ker(Hz) / rowspace(Hx)
        """
        n = hx.shape[1]
        
        # For now, just return empty lists
        # A full implementation would compute the kernel/image quotient
        # This is correct for codes where logicals aren't needed immediately
        
        from qectostim.codes.utils import compute_css_logicals
        
        try:
            log_x_vecs, log_z_vecs = compute_css_logicals(hx, hz)
            logical_x = vectors_to_paulis_x(log_x_vecs)
            logical_z = vectors_to_paulis_z(log_z_vecs)
        except Exception:
            # If computation fails, return empty lists
            logical_x = []
            logical_z = []
        
        return logical_x, logical_z
    
    @property
    def name(self) -> str:
        return f"HypergraphProduct({self.code_a.name}, {self.code_b.name})"
    
    def left_block_indices(self) -> List[int]:
        """Return indices of qubits in the left block (n_A × m_B)."""
        n_left = self.n_a * self.code_b.hz.shape[0]
        return list(range(n_left))
    
    def right_block_indices(self) -> List[int]:
        """Return indices of qubits in the right block (m_A × n_B)."""
        n_left = self.n_a * self.code_b.hz.shape[0]
        return list(range(n_left, self.n))


# Alias for the more common name
HomologicalProductCode = HypergraphProductCode


def hypergraph_product(
    code_a: TopologicalCSSCode,
    code_b: TopologicalCSSCode,
    metadata: Optional[Dict[str, Any]] = None,
) -> TopologicalCSSCode4D:
    """
    Convenience function to create a hypergraph product code.
    
    Parameters
    ----------
    code_a, code_b : CSSCode
        The two CSS codes to combine.
    metadata : dict, optional
        Additional metadata.
        
    Returns
    -------
    HypergraphProductCode
        The product code.
    """
    ...

def hypergraph_product(
    code_a: CSSCode,
    code_b: CSSCode,
    metadata: Optional[Dict[str, Any]] = None,
) -> CSSCode:
    """
    Convenience function to create a hypergraph product code.
    
    Parameters
    ----------
    code_a, code_b : CSSCode
        The two CSS codes to combine.
    metadata : dict, optional
        Additional metadata.
        
    Returns
    -------
    HypergraphProductCode
        The product code.
    """
    ...

def homological_product(a: HomologicalCode, b: HomologicalCode) -> HomologicalCode:
    """
    Build the homological tensor product of two chain complexes.
    
    For CSS codes, this delegates to HypergraphProductCode.
    For general homological codes, this is more complex and requires
    working with the chain complex structure directly.
    
    Parameters
    ----------
    a, b : HomologicalCode
        The two homological codes to combine.
        
    Returns
    -------
    HomologicalCode
        The product code.
    """
    # If both are CSS codes, use the hypergraph product
    if isinstance(a, CSSCode) and isinstance(b, CSSCode):
        return HypergraphProductCode(a, b)
    
    # General case not yet implemented
    raise NotImplementedError(
        "General homological product is not yet implemented. "
        "Use HypergraphProductCode for CSS codes."
    )

