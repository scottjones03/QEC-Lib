# src/qectostim/codes/composite/homological_product.py
r"""
Homological Product Codes
=========================

The **homological product** (also called the **hypergraph product** in its
simplest form) takes two CSS codes and combines them via the tensor product
of their underlying chain complexes.  This is *the* central algebraic
tool for building higher-dimensional topological codes from lower-dimensional
building blocks.

Background
----------
Every CSS code can be represented as a 2-term or longer chain complex of
vector spaces over GF(2).  If code *A* has a chain complex of length
:math:`n` and code *B* of length :math:`m`, their tensor product yields a
new chain complex of length :math:`n + m - 1`.  Physical qubits live on the
middle grade of this product complex, stabilisers on adjacent grades, and
meta-checks (if any) one grade further out.

Key identity at the chain level::

    (A ⊗ B)_k  =  ⊕_{i+j=k}  A_i ⊗ B_j
    ∂^{A⊗B}_k  =  ∂^A ⊗ I  +  (-1)^i · I ⊗ ∂^B   (signs mod 2 → just XOR)

The **Künneth formula** gives :math:`k_{\text{product}} = k_A \times k_B`,
and the distance satisfies :math:`d \ge \min(d_A, d_B)` (equality in the
balanced case).

Two entry-points
~~~~~~~~~~~~~~~~
1. **Tillich–Zémor HGP** (``TillichZemorHGP``): takes two *classical*
   parity-check matrices :math:`H_1, H_2` and produces a CSS code with
   :math:`n = n_1 n_2 + r_1 r_2` qubits and stabilisers built from
   Kronecker products.  This is the original "hypergraph product".

2. **Full homological product** (``HomologicalProductCode``): takes two
   ``CSSCodeWithComplex`` objects (any chain length) and computes the full
   tensor-product complex.  This generalises to products of toric codes,
   colour codes, etc.

Canonical examples
~~~~~~~~~~~~~~~~~~
- RepetitionCode (2-chain) :math:`\otimes` RepetitionCode → Toric code (3-chain)
- ToricCode (3-chain) :math:`\otimes` ToricCode → 4-D toric code (5-chain)

Distance analysis
~~~~~~~~~~~~~~~~~
For the Tillich–Zémor HGP with seed codes ``[n_i, k_i, d_i]``:

* **Lower bound**: :math:`d \ge \min(d_1, d_2)` (always)
* **Upper bound**: :math:`d \le \max(d_1, d_2)` (typical for balanced seeds)
* **Exact**: when both seeds are identical, :math:`d = d_1 = d_2`

Code Parameters
~~~~~~~~~~~~~~~
For two CSS seed codes ``[[n_a, k_a, d_a]]`` and ``[[n_b, k_b, d_b]]``:

**Full homological product** (``HomologicalProductCode``):

* **n** = dimension of the middle grade of the tensor-product complex
* **k** = ``k_a * k_b``  (Künneth formula)
* **d** ≥ ``min(d_a, d_b)``  (equality for balanced seed codes)

**Tillich–Zémor HGP** (``TillichZemorHGP``, from classical ``[n_i, k_i, d_i]`` seeds):

* **n** = ``n_1 * n_2 + r_1 * r_2``  where ``r_i`` = number of checks
* **k** = ``k_1 * k_2``
* **d** ≥ ``min(d_1, d_2)``

Stabiliser Structure
~~~~~~~~~~~~~~~~~~~~
Stabilisers are built from tensor products of the input chain complexes:

* **X stabilisers**: ``Hx = [H1 ⊗ I_{n2} | I_{r1} ⊗ H2^T]``  (for HGP)
  with weights determined by the row weights of the seed codes.
  Count = ``r_1 * n_2`` generators.
* **Z stabilisers**: ``Hz = [I_{n1} ⊗ H2 | H1^T ⊗ I_{r2}]``  (for HGP)
  with analogous structure.  Count = ``n_1 * r_2`` generators.
* **Meta-checks**: for chain complexes of length ≥ 5, additional
  meta-check layers arise from the grades adjacent to the stabiliser
  grades, enabling **single-shot error correction**.
* **Measurement schedule**: depends on the factor codes and target
  decoder.  For 5-chain codes, meta-checks may be measured in a
  separate round or interleaved with stabiliser measurements.

Decoder considerations
~~~~~~~~~~~~~~~~~~~~~~
Products of 3-chains or higher (chain length ≥ 5) introduce **hyperedge**
errors that cannot be decomposed into pair-wise edges.  Standard MWPM
decoders (PyMatching, Fusion Blossom) cannot handle these; use
BP-OSD or the Tesseract decoder instead.

Literature
----------
-  Tillich & Zémor, "Quantum LDPC codes with positive rate and minimum
   distance proportional to :math:`\sqrt n`", IEEE Trans. Inf. Theory 60
   (2), 2014. `arXiv:0903.0566 <https://arxiv.org/abs/0903.0566>`_
-  Bravyi & Hastings, "Homological product codes",
   `arXiv:1311.0885 <https://arxiv.org/abs/1311.0885>`_
-  Audoux & Couvreur, "On tensor products of CSS codes",
   `arXiv:1512.07081 <https://arxiv.org/abs/1512.07081>`_
-  Hastings, "Quantum codes from high-dimensional manifolds", 2016.

See Also
--------
qectostim.codes.small.repetition_codes : Simplest building block for HGP.
qectostim.codes.surface.toric_code : 3-chain code; its self-product gives
    a 4-D toric code.

Fault tolerance
---------------
* Products of distance-d codes yield codes with d_product ≥ d, so
  fault-tolerance properties are inherited from the seed codes.
* The meta-check structure arising from chain complexes of length ≥ 5
  enables single-shot error correction.

Implementation notes
--------------------
* The Kronecker product of sparse matrices is computed using
  ``scipy.sparse.kron`` for memory efficiency.
* Boundary operators are cached after the first computation to avoid
  redundant tensor-product calculations.
* For very large seed codes (n > 1000), the product code can exceed
  10⁶ qubits; users should verify memory availability before construction.
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
from qectostim.codes.utils import validate_css_code


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
    except Exception as exc:
        import warnings
        warnings.warn(
            f"Failed to compute CSS logicals: {exc!r}. "
            "Returning empty logical operator lists.",
            RuntimeWarning,
            stacklevel=2,
        )
        logical_x = []
        logical_z = []
    
    return logical_x, logical_z


class HomologicalProductCode(CSSCodeWithComplex):
    r"""Homological product of two CSS codes with chain complexes.

    Computes the full tensor product of the input chain complexes.  The
    resulting complex has:

    * **chain length** = ``len(A) + len(B) - 1``
    * **qubit grade** = middle grade = ``max_grade // 2``
    * **k** = :math:`k_A \times k_B`  (Künneth formula)
    * **d** ≥ :math:`\min(d_A, d_B)`

    For 3-chain ⊗ 3-chain (e.g. toric ⊗ toric) the result is a 5-chain
    complex whose qubits sit on grade 2.

    Parameters
    ----------
    code_a, code_b : CSSCodeWithComplex
        CSS codes whose ``chain_complex`` attributes are not ``None``.
    metadata : dict, optional
        Additional metadata merged into the auto-generated fields.

    Attributes
    ----------
    code_a, code_b : CSSCodeWithComplex
        The two factor codes.
    n_a, n_b, k_a, k_b : int
        Qubit / logical counts of the factor codes.

    Examples
    --------
    >>> from qectostim.codes.surface.toric_code import ToricCode33
    >>> t = ToricCode33()
    >>> prod = HomologicalProductCode(t, t)
    >>> print(prod.chain_length)  # 5
    >>> print(prod.k)             # k_A * k_B = 4

    Notes
    -----
    * Products of 3-chains (chain length ≥ 5) contain **hyperedge** errors.
      Standard MWPM decoders cannot decode them — use BP-OSD or Tesseract.
    * The ``stabiliser_schedule`` is set to ``None`` because the optimal
      measurement schedule depends on the specific factor codes and the
      target decoder.

    References
    ----------
    .. [1] Bravyi & Hastings, "Homological product codes",
       `arXiv:1311.0885 <https://arxiv.org/abs/1311.0885>`_
    .. [2] Audoux & Couvreur, "On tensor products of CSS codes",
       `arXiv:1512.07081 <https://arxiv.org/abs/1512.07081>`_

    See Also
    --------
    TillichZemorHGP : Simpler construction from classical parity-check
        matrices (2-chain inputs).
    homological_product : Factory function that dispatches to the right
        class.
    """
    
    def __init__(
        self,
        code_a: CSSCodeWithComplex,
        code_b: CSSCodeWithComplex,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Construct the homological product of two CSS codes.

        Computes the full tensor product of the input chain complexes
        and derives the stabiliser matrices, logical operators, and
        metadata for the resulting code.

        Parameters
        ----------
        code_a : CSSCodeWithComplex
            First CSS code (must have a non-``None`` ``chain_complex``).
        code_b : CSSCodeWithComplex
            Second CSS code (must have a non-``None`` ``chain_complex``).
        metadata : dict, optional
            Extra key/value pairs merged into the code's metadata dict.

        Raises
        ------
        ValueError
            If either input code has ``chain_complex is None``
            (incompatible structure for homological product).
        ValueError
            If the product check matrices fail CSS validation
            (``Hx @ Hz^T ≠ 0`` mod 2).
        """
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

        # ── Validate CSS structure ─────────────────────────────────
        validate_css_code(
            hx, hz,
            f"HomologicalProduct({code_a.name}, {code_b.name})",
            raise_on_error=True,
        )
        
        # Build metadata
        meta: Dict[str, Any] = dict(metadata or {})
        meta["code_a_name"] = code_a.name
        meta["code_b_name"] = code_b.name
        meta["construction"] = "chain_complex_tensor"
        meta["chain_length_a"] = cc_a.max_grade + 1
        meta["chain_length_b"] = cc_b.max_grade + 1
        meta["product_chain_length"] = product_complex.max_grade + 1
        meta["qubit_grade"] = qg
        
        # --- Standardised metadata -----------------------------------------
        meta["code_family"] = "homological_product"
        meta["code_type"] = "CSS"
        # Compute n and k from Hx/Hz
        n_product = hx.shape[1] if hx.size > 0 else (hz.shape[1] if hz.size > 0 else 0)
        k_product = self.k_a * self.k_b  # Künneth formula
        meta["n"] = n_product
        meta["k"] = k_product
        meta["rate"] = (k_product / n_product) if n_product > 0 else 0.0
        meta["distance"] = meta.get("distance", None)  # exact distance hard to compute
        meta["data_qubits"] = list(range(n_product))

        # Coordinate metadata
        cols_grid = int(np.ceil(np.sqrt(n_product)))
        data_coords_list = [(float(i % cols_grid), float(i // cols_grid)) for i in range(n_product)]
        meta.setdefault("data_coords", data_coords_list)
        x_stab_coords_list = []
        for row in hx:
            support = np.where(row)[0]
            if len(support) > 0:
                cx = float(np.mean([data_coords_list[q][0] for q in support if q < len(data_coords_list)]))
                cy = float(np.mean([data_coords_list[q][1] for q in support if q < len(data_coords_list)]))
                x_stab_coords_list.append((cx, cy))
            else:
                x_stab_coords_list.append((0.0, 0.0))
        meta.setdefault("x_stab_coords", x_stab_coords_list)
        z_stab_coords_list = []
        for row in hz:
            support = np.where(row)[0]
            if len(support) > 0:
                cx = float(np.mean([data_coords_list[q][0] for q in support if q < len(data_coords_list)]))
                cy = float(np.mean([data_coords_list[q][1] for q in support if q < len(data_coords_list)]))
                z_stab_coords_list.append((cx, cy))
            else:
                z_stab_coords_list.append((0.0, 0.0))
        meta.setdefault("z_stab_coords", z_stab_coords_list)

        # ── Logical operator Pauli types ───────────────────────
        meta["lx_pauli_type"] = "X"
        meta["lz_pauli_type"] = "Z"
        # ── Logical operator supports ──────────────────────────
        def _supp(op):
            if isinstance(op, str):
                return [i for i, c in enumerate(op) if c != 'I']
            elif isinstance(op, dict):
                return sorted(op.keys())
            return []
        if logical_x:
            meta["lx_support"] = [_supp(lx) for lx in logical_x] if len(logical_x) > 1 else _supp(logical_x[0])
        else:
            meta["lx_support"] = []
        if logical_z:
            meta["lz_support"] = [_supp(lz) for lz in logical_z] if len(logical_z) > 1 else _supp(logical_z[0])
        else:
            meta["lz_support"] = []

        # Homological product metadata for decoder compatibility
        chain_len = product_complex.max_grade + 1
        meta["is_homological_product"] = True
        meta["chain_length"] = chain_len
        # 5-chain and higher (4D+ codes) have hyperedge errors that can't be decomposed
        # Standard MWPM decoders (PyMatching, FusionBlossom) cannot handle these
        meta["has_hyperedges"] = chain_len >= 5
        meta["requires_hyperedge_decoder"] = chain_len >= 5
        meta["supports_standard_decoders"] = chain_len < 5

        # Stabiliser schedule: depends on factor codes and decoder
        meta["stabiliser_schedule"] = {
            "n_rounds": None,
            "description": (
                "Schedule depends on factor codes and target decoder.  "
                "For 5-chain codes, meta-checks enable single-shot "
                "decoding without repeated syndrome rounds."
            ),
        }
        meta["x_schedule"] = None  # depends on factor codes and target decoder
        meta["z_schedule"] = None  # depends on factor codes and target decoder

        # Literature
        meta["error_correction_zoo_url"] = (
            "https://errorcorrectionzoo.org/c/hypergraph_product"
        )
        meta["wikipedia_url"] = "https://en.wikipedia.org/wiki/Hypergraph_product_code"
        meta["canonical_references"] = [
            "Bravyi & Hastings, 'Homological product codes', arXiv:1311.0885",
            "Audoux & Couvreur, 'On tensor products of CSS codes', arXiv:1512.07081",
            "Tillich & Zémor, 'Quantum LDPC codes …', arXiv:0903.0566",
        ]
        meta["connections"] = [
            "Generalises hypergraph product (HGP) to arbitrary chain complexes",
            "Künneth formula: k_product = k_A × k_B",
            "Products of 3-chains yield 4-D codes with single-shot decoding",
            "Building block for fibre-bundle and balanced-product codes",
        ]
        
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
        """Human-readable name: ``HomologicalProduct(A, B)``."""
        return f"HomologicalProduct({self.code_a.name}, {self.code_b.name})"
    
    # =========================================================================
    # Decoder compatibility properties
    # =========================================================================
    @property
    def is_homological_product(self) -> bool:
        """True for homological product codes."""
        return self._metadata.get("is_homological_product", True)
    
    @property
    def chain_length(self) -> int:
        """Length of the product chain complex."""
        return self._metadata.get("chain_length", 5)
    
    @property
    def has_hyperedges(self) -> bool:
        """True if code has hyperedge errors (>2 detectors per error)."""
        return self._metadata.get("has_hyperedges", self.chain_length >= 5)
    
    @property
    def requires_hyperedge_decoder(self) -> bool:
        """True if code requires a hyperedge-capable decoder (BPOSD, Tesseract)."""
        return self._metadata.get("requires_hyperedge_decoder", self.has_hyperedges)
    
    @property
    def supports_standard_decoders(self) -> bool:
        """True if standard MWPM decoders (PyMatching, FusionBlossom) can handle this code."""
        return self._metadata.get("supports_standard_decoders", not self.has_hyperedges)


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
    r"""Standard Tillich–Zémor hypergraph product code from classical codes.

    Given two classical codes with parity-check matrices
    :math:`H_1` (:math:`r_1 \times n_1`) and :math:`H_2`
    (:math:`r_2 \times n_2`), the hypergraph product code has:

    * :math:`n = n_1 n_2 + r_1 r_2` physical qubits
    * :math:`k = k_1 k_2` logical qubits where :math:`k_i = n_i - r_i`
    * :math:`d \ge \min(d_1, d_2)`

    Stabiliser matrices::

        Hx = [ H1 ⊗ I_{n2}  |  I_{r1} ⊗ H2^T ]
        Hz = [ I_{n1} ⊗ H2   |  H1^T ⊗ I_{r2} ]

    Parameters
    ----------
    H1, H2 : np.ndarray
        Binary parity-check matrices of the two classical seed codes.
    metadata : dict, optional
        Additional metadata.

    Attributes
    ----------
    _H1, _H2 : np.ndarray
        Stored copies of the input parity-check matrices.

    Examples
    --------
    >>> import numpy as np
    >>> # [3,1,3] repetition code parity-check matrix
    >>> H = np.array([[1,1,0],[0,1,1]], dtype=np.uint8)
    >>> hgp = TillichZemorHGP(H, H)
    >>> print(f'[[{hgp.n}, {hgp.k}]]')  # [[13, 1]]

    Notes
    -----
    This is the original construction from Tillich & Zémor (2009).  When
    both seed codes are identical the product is **self-dual** (Hx and Hz
    have the same weight profile).

    References
    ----------
    .. [1] Tillich & Zémor, "Quantum LDPC codes with positive rate and
       minimum distance proportional to √n", arXiv:0903.0566.

    See Also
    --------
    HomologicalProductCode : General tensor product for arbitrary chain
        complexes.
    """
    
    def __init__(
        self,
        H1: np.ndarray,
        H2: np.ndarray,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Construct a Tillich–Zémor HGP from two classical parity-check matrices.

        Parameters
        ----------
        H1 : np.ndarray
            Binary parity-check matrix of the first classical seed code
            (shape ``(r1, n1)``).
        H2 : np.ndarray
            Binary parity-check matrix of the second classical seed code
            (shape ``(r2, n2)``).
        metadata : dict, optional
            Extra key/value pairs merged into the code's metadata dict.

        Raises
        ------
        ValueError
            If the product check matrices fail CSS validation
            (``Hx @ Hz^T ≠ 0`` mod 2).
        """
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

        # ── Validate CSS structure ─────────────────────────────────
        validate_css_code(
            hx, hz,
            f"TillichZemorHGP({H1.shape}, {H2.shape})",
            raise_on_error=True,
        )
        
        # Build metadata
        meta: Dict[str, Any] = dict(metadata or {})
        meta["code_family"] = "hypergraph_product"
        meta["code_type"] = "CSS"
        meta["construction"] = "tillich_zemor"
        meta["n1"] = n1
        meta["n2"] = n2
        meta["r1"] = r1
        meta["r2"] = r2
        meta["k1"] = n1 - r1
        meta["k2"] = n2 - r2
        total_n = n1 * n2 + r1 * r2
        k_product = (n1 - r1) * (n2 - r2)
        meta["n"] = total_n
        meta["k"] = k_product
        meta["rate"] = (k_product / total_n) if total_n > 0 else 0.0
        # Distance lower bound: d >= min(d1, d2)
        # Exact distance is hard to compute; store as None unless overridden
        meta["distance"] = meta.get("distance", None)
        meta["data_qubits"] = list(range(total_n))

        # Coordinate metadata
        cols_grid = int(np.ceil(np.sqrt(total_n)))
        data_coords_list = [(float(i % cols_grid), float(i // cols_grid)) for i in range(total_n)]
        meta.setdefault("data_coords", data_coords_list)
        x_stab_coords_list = []
        for row in hx:
            support = np.where(row)[0]
            if len(support) > 0:
                cx = float(np.mean([data_coords_list[q][0] for q in support if q < len(data_coords_list)]))
                cy = float(np.mean([data_coords_list[q][1] for q in support if q < len(data_coords_list)]))
                x_stab_coords_list.append((cx, cy))
            else:
                x_stab_coords_list.append((0.0, 0.0))
        meta.setdefault("x_stab_coords", x_stab_coords_list)
        z_stab_coords_list = []
        for row in hz:
            support = np.where(row)[0]
            if len(support) > 0:
                cx = float(np.mean([data_coords_list[q][0] for q in support if q < len(data_coords_list)]))
                cy = float(np.mean([data_coords_list[q][1] for q in support if q < len(data_coords_list)]))
                z_stab_coords_list.append((cx, cy))
            else:
                z_stab_coords_list.append((0.0, 0.0))
        meta.setdefault("z_stab_coords", z_stab_coords_list)

        # ── Logical operator Pauli types ───────────────────────
        meta["lx_pauli_type"] = "X"
        meta["lz_pauli_type"] = "Z"
        # ── Logical operator supports ──────────────────────────
        def _support_from_pauli(op):
            if isinstance(op, str):
                return [i for i, c in enumerate(op) if c != 'I']
            elif isinstance(op, dict):
                return sorted(op.keys())
            return []
        if logical_x:
            if len(logical_x) == 1:
                meta["lx_support"] = _support_from_pauli(logical_x[0])
            else:
                meta["lx_support"] = [_support_from_pauli(lx) for lx in logical_x]
        else:
            meta["lx_support"] = []
        if logical_z:
            if len(logical_z) == 1:
                meta["lz_support"] = _support_from_pauli(logical_z[0])
            else:
                meta["lz_support"] = [_support_from_pauli(lz) for lz in logical_z]
        else:
            meta["lz_support"] = []
        # ── Stabiliser scheduling ──────────────────────────────
        meta["stabiliser_schedule"] = {
            "n_rounds": None,
            "description": (
                "Schedule depends on factor-code layout and target "
                "decoder."
            ),
        }  # schedule depends on layout / decoder
        # NOTE: x_schedule/z_schedule set to None for product codes.
        # Optimal measurement scheduling depends on the specific factor codes
        # and the target decoder.
        meta["x_schedule"] = None
        meta["z_schedule"] = None
        # ── Literature / provenance ────────────────────────────
        meta["error_correction_zoo_url"] = (
            "https://errorcorrectionzoo.org/c/hypergraph_product"
        )
        meta["wikipedia_url"] = "https://en.wikipedia.org/wiki/Hypergraph_product_code"
        meta["canonical_references"] = [
            "Tillich & Zémor, 'Quantum LDPC codes …', arXiv:0903.0566",
        ]
        meta["connections"] = [
            "Special case of homological product with 2-chain inputs",
            "Self-product of [n,k,d] repetition code gives toric-like code",
            "Building block for fibre-bundle and lifted-product codes",
        ]
        
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
        """Human-readable name showing seed matrix shapes."""
        return f"TillichZemorHGP({self._H1.shape}, {self._H2.shape})"

    @property
    def distance(self) -> int:
        """Code distance (from metadata, or 1 if unknown).

        The exact distance of a hypergraph product code is hard to compute
        in general.  A lower bound ``min(d1, d2)`` is stored at construction
        if provided via metadata; otherwise returns 1 as a safe fallback.
        """
        d = self._metadata.get("distance")
        return d if d is not None else 1

    def qubit_coords(self) -> List[Tuple[float, float]]:
        """Return 2-D qubit coordinates based on the tensor structure.

        The first ``n1 * n2`` qubits are laid out on an ``n1 × n2`` grid.
        The remaining ``r1 * r2`` qubits are placed in an ``r1 × r2`` grid
        offset to the right.
        """
        r1, n1 = self._H1.shape
        _, n2 = self._H2.shape
        r2 = self._H2.shape[0]
        coords: List[Tuple[float, float]] = []
        # Left sector: n1 x n2
        for i in range(n1 * n2):
            coords.append((float(i % n2), float(i // n2)))
        # Right sector: r1 x r2, offset
        x_off = float(n2 + 1)
        for i in range(r1 * r2):
            coords.append((x_off + float(i % r2), float(i // r2)))
        return coords
