r"""
4-D Toric Code (Tesseract Code)
================================

The **4-D toric code** is constructed by taking two successive homological
tensor products of the 1-D repetition-code chain complex::

    rep  (2-chain)  ⊗  rep  →  toric  (3-chain)
    toric (3-chain)  ⊗  toric  →  4-D code  (5-chain)

The resulting 5-chain complex

.. math::

    C_4 \xrightarrow{\partial_4}
    C_3 \xrightarrow{\partial_3}
    C_2 \xrightarrow{\partial_2}
    C_1 \xrightarrow{\partial_1}
    C_0

places **qubits on grade 2** (faces), X-stabilisers on grade 3 (cells),
Z-stabilisers on grade 1 (edges), and **meta-checks** on grades 4 and 0.
The meta-checks enable **single-shot** quantum error correction.

Code parameters
~~~~~~~~~~~~~~~
On an :math:`L \\times L \\times L \\times L` torus:

+-----+-----------+-----+------+
|  L  |     n     |  k  |  d   |
+=====+===========+=====+======+
|  2  |    96     |  6  |  4   |
+-----+-----------+-----+------+
|  3  |   486     |  6  |  9   |
+-----+-----------+-----+------+
|  4  |  1536     |  6  | 16   |
+-----+-----------+-----+------+

where :math:`n = 6L^4`, :math:`k = \\binom{4}{2} = 6`,
:math:`d = L^2`.

Key features
~~~~~~~~~~~~
- **Single-shot decoding**: meta-checks (:math:`\\partial_1` on Z-syndrome,
  :math:`\\partial_4^T` on X-syndrome) allow syndrome noise to be corrected
  without repeated measurement rounds.
- **Hyperedge errors**: errors can trigger ≥ 3 detectors; standard MWPM
  decoders cannot handle them.  Use BP-OSD, Union-Find, or the Tesseract
  decoder.
- Equivalent to ``HomologicalProductCode(toric, toric)`` but with explicit
  4-D cell labelling and coordinates.

References
----------
.. [1] Dennis, Kitaev, Landahl & Preskill, "Topological quantum memory",
   J. Math. Phys. 43, 4452 (2002).  arXiv:quant-ph/0110143
.. [2] Breuckmann & Ni, "Scalable Neural Network Decoders for Higher
   Dimensional Quantum Codes", Quantum 2, 68 (2018).
.. [3] Error Correction Zoo — 4D surface code,
   https://errorcorrectionzoo.org/c/surface
.. [4] Bravyi & Hastings, "Homological product codes",
   arXiv:1311.0885

See Also
--------
qectostim.codes.composite.homological_product : General tensor-product
    construction (can build this code as ``HomologicalProductCode(toric, toric)``).
qectostim.codes.surface.toric_3d : 3-D toric code (one tensor-product
    level below).
"""
from __future__ import annotations
from typing import Dict, Any, List, Optional, Tuple

import numpy as np

from qectostim.codes.abstract_css import TopologicalCSSCode4D
from qectostim.codes.abstract_code import PauliString
from qectostim.codes.complexes.chain_complex import ChainComplex, tensor_product_chain_complex
from qectostim.codes.complexes.css_complex import FiveCSSChainComplex
from qectostim.codes.utils import compute_css_logicals, vectors_to_paulis_x, vectors_to_paulis_z, validate_css_code


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _gf2_rank(M: np.ndarray) -> int:
    """Compute matrix rank over GF(2) via row-echelon reduction."""
    M = M.copy().astype(np.int64) % 2
    rows, cols = M.shape
    r = 0
    for c in range(cols):
        pivot = None
        for row in range(r, rows):
            if M[row, c]:
                pivot = row
                break
        if pivot is None:
            continue
        M[[r, pivot]] = M[[pivot, r]]
        for row in range(rows):
            if row != r and M[row, c]:
                M[row] = (M[row] + M[r]) % 2
        r += 1
    return r


def _build_rep_complex(L: int) -> ChainComplex:
    """Build 2-chain complex for L-qubit periodic repetition code.

    Returns a :class:`ChainComplex` with a single boundary map
    ``∂₁ : C₁ → C₀`` of shape ``(L, L)`` (periodic / toric boundary
    conditions).
    """
    boundary_1 = np.zeros((L, L), dtype=np.uint8)
    for v in range(L):
        boundary_1[v, v] = 1
        boundary_1[v, (v + 1) % L] = 1
    return ChainComplex(boundary_maps={1: boundary_1}, qubit_grade=1)


def _build_4d_complex(L: int) -> ChainComplex:
    """Build the 5-chain complex for the 4-D toric code with side *L*.

    Steps
    -----
    1. ``rep = _build_rep_complex(L)``  — 2-chain (C₁ → C₀)
    2. ``toric = rep ⊗ rep``           — 3-chain (C₂ → C₁ → C₀)
    3. ``code4d = toric ⊗ toric``      — 5-chain (C₄ → C₃ → C₂ → C₁ → C₀)

    Returns
    -------
    ChainComplex
        qubit_grade = 2, boundary maps at grades 1–4.
    """
    rep = _build_rep_complex(L)
    toric = tensor_product_chain_complex(rep, rep)
    return tensor_product_chain_complex(toric, toric)


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class FourDSurfaceCode(TopologicalCSSCode4D):
    r"""4-D toric (tesseract) code on an L⁴ hyper-torus.

    Built by two levels of tensor product:
    ``rep(L) ⊗ rep(L) → toric(L)``  then  ``toric(L) ⊗ toric(L) → 4D(L)``.

    Parameters
    ----------
    L : int, optional
        Linear lattice size (default 2).  Must be ≥ 2.
    metadata : dict, optional
        Additional metadata merged into the code's metadata dict.

    Attributes
    ----------
    n : int
        Number of physical qubits (= 6 L⁴).
    k : int
        Number of logical qubits (= 6).
    distance : int
        Code distance (= L²).
    hx : np.ndarray
        X-stabiliser parity-check matrix (from ∂₃ᵀ).
    hz : np.ndarray
        Z-stabiliser parity-check matrix (from ∂₂).
    meta_x : np.ndarray
        Z-syndrome meta-checks (from ∂₁).
    meta_z : np.ndarray
        X-syndrome meta-checks (from ∂₄ᵀ).

    Examples
    --------
    >>> code = FourDSurfaceCode(L=2)
    >>> code.n, code.k, code.distance
    (96, 6, 4)
    >>> code.has_metachecks
    True

    Notes
    -----
    - The code has **hyperedge** errors that standard MWPM decoders
      cannot handle; use BP-OSD, Union-Find, or Tesseract decoder.
    - Meta-checks enable **single-shot** error correction without
      repeated syndrome measurement.

    References
    ----------
    .. [1] Dennis et al., "Topological quantum memory",
       J. Math. Phys. 43, 4452 (2002).  arXiv:quant-ph/0110143
    .. [2] Bravyi & Hastings, "Homological product codes",
       arXiv:1311.0885

    See Also
    --------
    HomologicalProductCode :
        Algebraic route to the same code via ``HomologicalProductCode(toric, toric)``.
    ToricCode3D :
        3-D toric code (one tensor-product level below).
    """

    def __init__(
        self,
        L: int = 2,
        *,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        if L < 2:
            raise ValueError(f"L must be >= 2, got {L}")
        self._L = L

        # ── Build chain complex ────────────────────────────────────
        cc_raw = _build_4d_complex(L)

        # Wrap into FiveCSSChainComplex
        sigma4 = cc_raw.boundary_maps[4].astype(np.uint8)
        sigma3 = cc_raw.boundary_maps[3].astype(np.uint8)
        sigma2 = cc_raw.boundary_maps[2].astype(np.uint8)
        sigma1 = cc_raw.boundary_maps[1].astype(np.uint8)

        chain_complex = FiveCSSChainComplex(
            sigma4=sigma4, sigma3=sigma3, sigma2=sigma2, sigma1=sigma1,
            qubit_grade=2,
        )

        # ── Parity-check matrices ─────────────────────────────────
        hx = chain_complex.hx   # ∂₃ᵀ
        hz = chain_complex.hz   # ∂₂
        n = hx.shape[1]

        # ── Code parameters (GF(2) rank) ──────────────────────────
        r_hx = _gf2_rank(hx)
        r_hz = _gf2_rank(hz)
        k = n - r_hx - r_hz
        d = L * L

        # ── Logical operators ──────────────────────────────────────
        lx_vecs, lz_vecs = compute_css_logicals(hx, hz)
        lx_vecs = lx_vecs[:k]
        lz_vecs = lz_vecs[:k]
        logical_x: List[PauliString] = vectors_to_paulis_x(lx_vecs)
        logical_z: List[PauliString] = vectors_to_paulis_z(lz_vecs)

        # ── Validate CSS structure ─────────────────────────────────
        validate_css_code(hx, hz, f"FourDSurfaceCode_L{L}", raise_on_error=True)

        # ── Logical supports ──────────────────────────────────────
        def _support(v: np.ndarray) -> List[int]:
            return [int(i) for i in np.nonzero(v)[0]]

        if k == 1:
            lx_support = _support(lx_vecs[0])
            lz_support = _support(lz_vecs[0])
        else:
            lx_support = [_support(v) for v in lx_vecs]
            lz_support = [_support(v) for v in lz_vecs]

        # ── Metadata ───────────────────────────────────────────────
        meta: Dict[str, Any] = dict(metadata or {})
        meta.update(
            {
                # 17 standard keys
                "code_family": "surface",
                "code_type": "4d_toric",
                "distance": d,
                "n": n,
                "k": k,
                "rate": k / n if n > 0 else 0.0,
                "lx_pauli_type": "X",
                "lz_pauli_type": "Z",
                "lx_support": lx_support,
                "lz_support": lz_support,
                "data_qubits": list(range(n)),
                "stabiliser_schedule": {
                    "n_rounds": None,
                    "description": (
                        "Schedule depends on 4D geometry and target "
                        "decoder.  Meta-checks enable single-shot "
                        "decoding without repeated syndrome rounds."
                    ),
                },
                "x_schedule": None,
                "z_schedule": None,
                "error_correction_zoo_url": (
                    "https://errorcorrectionzoo.org/c/surface"
                ),
                "wikipedia_url": (
                    "https://en.wikipedia.org/wiki/Toric_code"
                ),
                "canonical_references": [
                    "Dennis, Kitaev, Landahl & Preskill, "
                    "J. Math. Phys. 43, 4452 (2002). arXiv:quant-ph/0110143",
                    "Bravyi & Hastings, 'Homological product codes', "
                    "arXiv:1311.0885",
                ],
                "connections": [
                    "rep ⊗ rep → toric, toric ⊗ toric → 4D toric",
                    "Equivalent to HomologicalProductCode(toric, toric)",
                    "Single-shot error correction via meta-checks",
                    "Hyperedge errors: requires BP-OSD / Union-Find decoder",
                    "k = C(4,2) = 6 logical qubits on the 4-torus",
                ],
                # Extra metadata
                "boundary_conditions": "periodic (4-torus)",
                "lattice_size": L,
                "dimension": 4,
                "chain_length": 5,
                "has_metachecks": True,
                "has_hyperedges": True,
                "requires_hyperedge_decoder": True,
                "supports_standard_decoders": False,
            }
        )

        # ── Initialise TopologicalCSSCode4D ────────────────────────
        super().__init__(
            chain_complex=chain_complex,
            logical_x=logical_x,
            logical_z=logical_z,
            metadata=meta,
        )

        # Override Hx / Hz to ensure binary
        self._hx = hx.astype(np.uint8) % 2
        self._hz = hz.astype(np.uint8) % 2
        self._d = d

    # ── Properties ─────────────────────────────────────────────────

    @property
    def distance(self) -> int:
        """Code distance (= L² for the 4-D toric code)."""
        return self._d

    @property
    def name(self) -> str:
        """Human-readable name, e.g. ``'FourDSurfaceCode(L=2)'``."""
        return f"FourDSurfaceCode(L={self._L})"

    @property
    def lattice_size(self) -> int:
        """Lattice side length *L*."""
        return self._L

    def qubit_coords(self) -> Optional[List[Tuple[float, ...]]]:
        """Return 4D coordinates for qubits (None — no embedding yet)."""
        return None
