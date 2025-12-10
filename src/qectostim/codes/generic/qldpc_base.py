# src/qectostim/codes/generic/qldpc_base.py
"""
Base classes for Quantum Low-Density Parity-Check (QLDPC) codes.

QLDPC codes are CSS codes where both Hx and Hz are sparse (low-density),
meaning each stabilizer has bounded weight and each qubit participates
in a bounded number of checks.

These codes are algebraic (no geometric embedding) but still have
the CSS structure from chain complexes.

Class Hierarchy:
    CSSCode
    └── QLDPCCode - Base for algebraic QLDPC codes
        ├── HypergraphProductCode - HGP codes from classical LDPC
        ├── BivariateBicycleCode - Cyclic QLDPC codes
        ├── LiftedProductCode - Lifted product codes
        ├── BalancedProductCode - Balanced product codes
        └── FiberBundleCode - Fiber bundle codes

Key properties of QLDPC codes:
- Sparse parity check matrices (LDPC-like)
- Good rate-distance tradeoffs
- Typically no geometric embedding (algebraic construction)
- chain_length = 3 (same as 2D topological codes)
"""
from __future__ import annotations
from abc import ABC
from typing import Dict, Any, List, Optional, Tuple

import numpy as np

from qectostim.codes.abstract_css import CSSCode
from qectostim.codes.abstract_code import PauliString


class QLDPCCode(CSSCode):
    """
    Base class for Quantum Low-Density Parity-Check codes.
    
    QLDPC codes have sparse stabilizer generators, meaning:
    - Each stabilizer has weight at most w (row weight)
    - Each qubit is in at most c checks (column weight)
    
    Unlike TopologicalCSSCode, QLDPC codes are algebraically constructed
    and don't have a natural geometric embedding, though they still
    have chain_length = 3 (3-chain structure).
    
    Subclasses implement specific QLDPC constructions:
    - Hypergraph products
    - Bivariate bicycle codes
    - Lifted products
    - Balanced products
    - Fiber bundles
    
    Properties
    ----------
    row_weight : int
        Maximum weight of any stabilizer
    column_weight : int
        Maximum number of stabilizers any qubit participates in
    rate : float
        Asymptotic encoding rate k/n
    """

    def __init__(
        self,
        hx: np.ndarray,
        hz: np.ndarray,
        logical_x: List[PauliString],
        logical_z: List[PauliString],
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize a QLDPC code.
        
        Parameters
        ----------
        hx : np.ndarray
            X-stabilizer parity check matrix
        hz : np.ndarray
            Z-stabilizer parity check matrix
        logical_x : List[PauliString]
            Logical X operators
        logical_z : List[PauliString]
            Logical Z operators
        metadata : dict, optional
            Additional metadata
        """
        meta = dict(metadata or {})
        meta["is_qldpc"] = True
        meta.setdefault("chain_length", 3)  # QLDPC are 3-chain
        
        super().__init__(
            hx=hx,
            hz=hz,
            logical_x=logical_x,
            logical_z=logical_z,
            metadata=meta,
        )

    @property
    def row_weight(self) -> int:
        """Maximum stabilizer weight (row weight of Hx or Hz)."""
        max_x = np.max(np.sum(self._hx, axis=1)) if self._hx.size > 0 else 0
        max_z = np.max(np.sum(self._hz, axis=1)) if self._hz.size > 0 else 0
        return int(max(max_x, max_z))

    @property
    def column_weight(self) -> int:
        """Maximum number of checks per qubit (column weight)."""
        col_x = np.max(np.sum(self._hx, axis=0)) if self._hx.size > 0 else 0
        col_z = np.max(np.sum(self._hz, axis=0)) if self._hz.size > 0 else 0
        return int(max(col_x, col_z))

    @property
    def rate(self) -> float:
        """Encoding rate k/n."""
        if self.n == 0:
            return 0.0
        return self.k / self.n

    @property
    def is_ldpc(self) -> bool:
        """Check if both row and column weights are O(1) (bounded)."""
        # Consider LDPC if weights are <= 10
        return self.row_weight <= 10 and self.column_weight <= 10

    def qubit_coords(self) -> Optional[List[Tuple[float, float]]]:
        """
        QLDPC codes typically don't have geometric embedding.
        
        Returns None by default. Subclasses can override if they
        have a natural layout (e.g., for visualization).
        """
        return None


class HypergraphProductBase(QLDPCCode):
    """
    Base class for Hypergraph Product codes.
    
    HGP codes are constructed from two classical LDPC codes via:
    
        Hx = [H1 ⊗ I_n2,  I_r1 ⊗ H2^T]
        Hz = [I_n1 ⊗ H2,  H1^T ⊗ I_r2]
    
    where H1 is r1×n1 and H2 is r2×n2.
    
    Properties:
    - n = n1*n2 + r1*r2 qubits
    - k = k1*k2 logical qubits
    - d = min(d1, d2) distance
    """

    def __init__(
        self,
        H1: np.ndarray,
        H2: np.ndarray,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Construct HGP code from two classical parity check matrices.
        
        Parameters
        ----------
        H1 : np.ndarray
            First classical code parity check matrix (r1 × n1)
        H2 : np.ndarray
            Second classical code parity check matrix (r2 × n2)
        metadata : dict, optional
            Additional metadata
        """
        self._H1 = np.array(H1, dtype=np.uint8) % 2
        self._H2 = np.array(H2, dtype=np.uint8) % 2
        
        hx, hz = self._build_hgp_matrices()
        n_qubits = hx.shape[1]
        
        # Compute k from ranks
        rank_hx = int(np.linalg.matrix_rank(hx.astype(float) % 2))
        rank_hz = int(np.linalg.matrix_rank(hz.astype(float) % 2))
        k = n_qubits - rank_hx - rank_hz
        
        # Build placeholder logical operators
        logical_x, logical_z = self._build_logicals(n_qubits, max(1, k))
        
        meta = dict(metadata or {})
        meta.update({
            "H1_shape": self._H1.shape,
            "H2_shape": self._H2.shape,
            "construction": "hypergraph_product",
        })
        
        super().__init__(
            hx=hx,
            hz=hz,
            logical_x=logical_x,
            logical_z=logical_z,
            metadata=meta,
        )

    def _build_hgp_matrices(self) -> Tuple[np.ndarray, np.ndarray]:
        """Build Hx and Hz from the HGP construction."""
        r1, n1 = self._H1.shape
        r2, n2 = self._H2.shape
        
        # Hx = [H1 ⊗ I_n2 | I_r1 ⊗ H2^T]
        hx_left = np.kron(self._H1, np.eye(n2, dtype=np.uint8))
        hx_right = np.kron(np.eye(r1, dtype=np.uint8), self._H2.T)
        hx = np.hstack([hx_left, hx_right]) % 2
        
        # Hz = [I_n1 ⊗ H2 | H1^T ⊗ I_r2]
        hz_left = np.kron(np.eye(n1, dtype=np.uint8), self._H2)
        hz_right = np.kron(self._H1.T, np.eye(r2, dtype=np.uint8))
        hz = np.hstack([hz_left, hz_right]) % 2
        
        return hx.astype(np.uint8), hz.astype(np.uint8)

    @staticmethod
    def _build_logicals(n: int, k: int) -> Tuple[List[str], List[str]]:
        """Build placeholder logical operators."""
        logical_x = []
        logical_z = []
        for i in range(k):
            lx = ['I'] * n
            lz = ['I'] * n
            if i < n:
                lx[i] = 'X'
                lz[i] = 'Z'
            logical_x.append(''.join(lx))
            logical_z.append(''.join(lz))
        return logical_x, logical_z


class BivariateBicycleBase(QLDPCCode):
    """
    Base class for Bivariate Bicycle codes.
    
    BB codes use circulant matrices derived from polynomial pairs:
    
        Hx = [A | B]
        Hz = [B^T | A^T]
    
    where A = x^a1 + x^a2 + ..., B = x^b1 + x^b2 + ... as circulants.
    
    Properties:
    - Highly structured (circulant blocks)
    - Good for hardware implementation
    - Include the [[144,12,12]] Gross code
    """
    pass


class BalancedProductBase(QLDPCCode):
    """
    Base class for Balanced Product codes.
    
    Balanced products are a generalization of HGP codes with
    additional group structure for improved parameters.
    """
    pass


class FiberBundleBase(QLDPCCode):
    """
    Base class for Fiber Bundle codes.
    
    Fiber bundle codes are constructed from a base code and fiber,
    generalizing the HGP construction with more flexible structure.
    """
    pass


class LiftedProductBase(QLDPCCode):
    """
    Base class for Lifted Product codes.
    
    Lifted product codes use group-structured lifting to achieve
    better rate-distance tradeoffs than standard HGP.
    """
    pass
