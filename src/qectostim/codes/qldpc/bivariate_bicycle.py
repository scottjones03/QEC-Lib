"""Bivariate Bicycle (BB) Codes

Bivariate Bicycle codes are a family of **QLDPC** codes constructed from
circulant matrices over the group ``ℤ_l × ℤ_m``.  They were proposed by
IBM as practical near-term candidates for fault-tolerant quantum error
correction and have since become a benchmark family.

Construction
------------
Given polynomials over ``ℤ_l × ℤ_m``:

.. math::

   A(x, y) = \\sum x^{a_i} y^{b_i}, \\quad
   B(x, y) = \\sum x^{c_j} y^{d_j}

the parity-check matrices are:

    Hx = [A | B]       (block_size × 2·block_size)
    Hz = [Bᵀ | Aᵀ]    (block_size × 2·block_size)

where ``block_size = l · m`` and A, B are ``block_size × block_size``
circulant matrices.  The CSS condition ``Hx · Hzᵀ = 0`` follows from
the identity ``A Bᵀ + B Aᵀ = 0  (mod 2)`` which holds because
circulant multiplication is commutative over GF(2).

Code parameters
---------------
* **n** = 2 · l · m   physical qubits (two blocks)
* **k** = n − rank(Hx) − rank(Hz)
* **d** depends on the polynomial choice — computed or looked up
* **Row weight** = |A_terms| + |B_terms| per row
* **Rate** R = k / n  (can approach a constant)

Stabiliser Structure
~~~~~~~~~~~~~~~~~~~~
* Each X- and Z-stabiliser has weight ``|A_terms| + |B_terms|``
  (typically 6 for the [[144,12,12]] Gross code).
* Total stabiliser count: ``l · m`` X-stabilisers and ``l · m``
  Z-stabilisers (before removing linear dependencies).
* All stabilisers can be measured in a single round because the
  low-density structure limits qubit contention; a greedy colouring
  of the Tanner graph gives O(1) parallel rounds.

Notable instances
-----------------
* **[[144, 12, 12]] Gross code**: ``l = 12, m = 6``,
  ``A = 1 + x³ + x⁶ + x⁹``, ``B = y + y² + x⁴y + x⁸y²``.
  One of the best-known BB codes.

Decoding
--------
BB codes are LDPC so belief-propagation with ordered-statistics
post-processing (BP+OSD) achieves near-optimal decoding.  The
circulant structure can also be exploited for FFT-accelerated BP.

Syndrome extraction
-------------------
The low weight of each row of ``Hx`` / ``Hz`` (equal to
``|A_terms| + |B_terms|``) means each stabiliser measurement
requires only a constant number of CNOT gates, making BB codes
attractive for near-term hardware with limited connectivity.

Connections
-----------
* Generalised bicycle (GB) codes are the univariate ``(m = 1)`` case.
* Hypergraph product codes are a special limit.
* Lifted product codes generalise the construction to non-abelian groups.
* Quantum Tanner codes share the algebraic-graph flavour.

References
----------
* Bravyi, Cross, Gambetta, Maslov, Rall & Yoder, "High-threshold and
  low-overhead fault-tolerant quantum memory", Nature **627**, 778–782
  (2024).  arXiv:2308.07915
* Panteleev & Kalachev, "Degenerate quantum LDPC codes with good
  finite length performance", Quantum **5**, 585 (2021).
  arXiv:1904.02703
* Error Correction Zoo: https://errorcorrectionzoo.org/c/qcga

Fault tolerance
---------------
* Constant stabiliser weight means syndrome circuits have O(1) depth,
  independent of block size — ideal for scalable architectures.
* The [[144,12,12]] Gross code achieves a 1.2 % threshold under
  circuit-level depolarising noise with BP+OSD decoding.
* Logical CNOT can be performed transversally between two identical
  BB code blocks.

Implementation notes
--------------------
* Circulant matrices are stored as 1-D coefficient vectors and
  expanded to full form only when building the Tanner graph.
* Polynomial multiplication modulo (x^l − 1, y^m − 1) is implemented
  via index arithmetic rather than symbolic algebra.
* Automorphisms of the ℤ_l × ℤ_m group can be used to search for
  equivalent codes and prune the parameter space.
* The ``validate_css_code`` utility is called at construction time to
  verify Hx · Hz^T = 0 (mod 2).
"""

from __future__ import annotations
from typing import Dict, Any, List, Optional, Tuple, Set

import warnings
import numpy as np

from qectostim.codes.generic.qldpc_base import QLDPCCode
from qectostim.codes.abstract_code import PauliString
from qectostim.codes.utils import compute_css_logicals, vectors_to_paulis_x, vectors_to_paulis_z, validate_css_code


def _circulant_from_polynomial(l: int, m: int, terms: List[Tuple[int, int]]) -> np.ndarray:
    """Build circulant matrix from polynomial terms over Z_l × Z_m.
    
    Parameters
    ----------
    l : int
        Size of first cyclic group
    m : int  
        Size of second cyclic group
    terms : List[Tuple[int, int]]
        List of (a, b) pairs representing x^a y^b terms
        
    Returns
    -------
    np.ndarray
        (l*m) × (l*m) circulant matrix
    """
    n = l * m
    matrix = np.zeros((n, n), dtype=np.uint8)
    
    for row in range(n):
        # Row index corresponds to (i, j) where row = i*m + j
        i, j = row // m, row % m
        
        for a, b in terms:
            # The column for term x^a y^b at position (i,j) is ((i+a) mod l, (j+b) mod m)
            col_i = (i + a) % l
            col_j = (j + b) % m
            col = col_i * m + col_j
            matrix[row, col] = (matrix[row, col] + 1) % 2
    
    return matrix


class BivariateBicycleCode(QLDPCCode):
    """Bivariate Bicycle QLDPC code over ``ℤ_l × ℤ_m``.

    Constructs a CSS code from two polynomials ``A(x, y)`` and
    ``B(x, y)`` whose circulant representations give parity-check
    matrices with constant row weight.

    Parameters
    ----------
    l : int
        Size of the first cyclic group (ℤ_l).
    m : int
        Size of the second cyclic group (ℤ_m).
    A_terms : list of (int, int)
        Exponent pairs ``(a_i, b_i)`` for ``A(x,y) = Σ x^{a_i} y^{b_i}``.
    B_terms : list of (int, int)
        Exponent pairs ``(c_j, d_j)`` for ``B(x,y) = Σ x^{c_j} y^{d_j}``.
    metadata : dict, optional
        Extra key/value pairs merged into the code's metadata dictionary.

    Attributes
    ----------
    n : int
        Number of physical qubits (``2 · l · m``).
    k : int
        Number of logical qubits.
    hx : np.ndarray
        X-stabiliser parity-check matrix ``[A | B]``.
    hz : np.ndarray
        Z-stabiliser parity-check matrix ``[Bᵀ | Aᵀ]``.

    Examples
    --------
    >>> code = BivariateBicycleCode(12, 6,
    ...     [(0,0),(3,0),(6,0),(9,0)],
    ...     [(0,1),(0,2),(4,1),(8,2)])
    >>> code.n, code.k
    (144, 12)

    Notes
    -----
    The CSS condition ``Hx · Hzᵀ = 0`` is guaranteed by the
    commutativity of circulant multiplication over GF(2).

    See Also
    --------
    GeneralizedBicycleCode : Univariate ``(m = 1)`` special case.
    LiftedProductCode     : Non-abelian generalisation.
    """

    def __init__(
        self,
        l: int,
        m: int,
        A_terms: List[Tuple[int, int]],
        B_terms: List[Tuple[int, int]],
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Initialise a bivariate bicycle code.

        Builds circulant matrices from the polynomial terms, constructs
        ``Hx = [A | B]`` and ``Hz = [Bᵀ | Aᵀ]``, computes logical
        operators, and populates all standard metadata fields.

        Parameters
        ----------
        l : int
            Size of ℤ_l.
        m : int
            Size of ℤ_m.
        A_terms : list of (int, int)
            Polynomial exponent pairs for A.
        B_terms : list of (int, int)
            Polynomial exponent pairs for B.
        metadata : dict, optional
            Extra metadata merged into the code's metadata dictionary.

        Raises
        ------
        ValueError
            If ``l < 2`` or ``m < 2``.
        ValueError
            If the constructed matrices violate the CSS condition
            ``Hx · Hz^T ≠ 0 (mod 2)``.
        """
        if l < 2:
            raise ValueError(f"l must be at least 2, got {l}")
        if m < 2:
            raise ValueError(f"m must be at least 2, got {m}")

        # Build circulant matrices
        A = _circulant_from_polynomial(l, m, A_terms)
        B = _circulant_from_polynomial(l, m, B_terms)
        
        # Total qubits: 2 * l * m (two blocks)
        block_size = l * m
        n_qubits = 2 * block_size
        
        # Hx = [A | B]
        hx = np.hstack([A, B]) % 2
        
        # Hz = [B^T | A^T]
        hz = np.hstack([B.T, A.T]) % 2
        
        # Verify CSS condition: Hx @ Hz.T = 0
        # For BB codes: A @ B^T + B @ A^T = 0 (mod 2) by construction
        comm = (hx @ hz.T) % 2
        if np.any(comm):
            raise ValueError("BB code construction failed: Hx Hz^T != 0")
        
        # Compute k
        rank_hx = np.linalg.matrix_rank(hx)
        rank_hz = np.linalg.matrix_rank(hz)
        k = n_qubits - rank_hx - rank_hz
        
        # Compute proper logical operators using CSS kernel/image prescription
        try:
            log_x_vecs, log_z_vecs = compute_css_logicals(hx, hz)
            logical_x = vectors_to_paulis_x(log_x_vecs) if log_x_vecs else [{0: 'X'}]
            logical_z = vectors_to_paulis_z(log_z_vecs) if log_z_vecs else [{0: 'Z'}]
        except Exception as e:
            # Fallback to single-qubit placeholder if computation fails
            warnings.warn(
                f"BivariateBicycleCode: logical operator computation failed "
                f"({e}); falling back to single-qubit placeholder."
            )
            logical_x = [{0: 'X'}]
            logical_z = [{0: 'Z'}]
        
        # ═══════════════════════════════════════════════════════════════════
        # METADATA (all 17 standard keys)
        # ═══════════════════════════════════════════════════════════════════
        # Compute logical support from first logical pair
        lx0_support = sorted(logical_x[0].keys()) if isinstance(logical_x[0], dict) else []
        lz0_support = sorted(logical_z[0].keys()) if isinstance(logical_z[0], dict) else []

        meta = dict(metadata or {})
        meta["code_family"] = "qldpc"
        meta["code_type"] = "bivariate_bicycle"
        meta["name"] = f"BB_{l}x{m}"
        meta["n"] = n_qubits
        meta["k"] = k
        meta["distance"] = None  # must be computed externally or looked up
        meta["rate"] = float(k) / n_qubits if n_qubits > 0 else 0.0
        meta["l"] = l
        meta["m"] = m
        meta["A_terms"] = A_terms
        meta["B_terms"] = B_terms
        meta["is_qldpc"] = True
        meta["row_weight"] = len(A_terms) + len(B_terms)

        meta["lx_pauli_type"] = "X"
        meta["lx_support"] = lx0_support
        meta["lz_pauli_type"] = "Z"
        meta["lz_support"] = lz0_support

        # Grid coordinates — two l×m blocks side by side
        data_coords = []
        for idx in range(block_size):
            i, j = idx // m, idx % m
            data_coords.append((float(i), float(j)))
        for idx in range(block_size):
            i, j = idx // m, idx % m
            data_coords.append((float(i + l + 1), float(j)))
        meta["data_coords"] = data_coords
        meta["data_qubits"] = list(range(n_qubits))
        meta["x_logical_coords"] = [data_coords[q] for q in lx0_support] if lx0_support else []
        meta["z_logical_coords"] = [data_coords[q] for q in lz0_support] if lz0_support else []

        # ── Stabiliser centroid coordinates ────────────────────────
        x_stab_coords_list = []
        for row_idx in range(hx.shape[0]):
            support = np.where(hx[row_idx])[0]
            if len(support) > 0:
                cx = np.mean([data_coords[q][0] for q in support])
                cy = np.mean([data_coords[q][1] for q in support])
                x_stab_coords_list.append((float(cx), float(cy)))
            else:
                x_stab_coords_list.append((0.0, 0.0))
        meta["x_stab_coords"] = x_stab_coords_list

        z_stab_coords_list = []
        for row_idx in range(hz.shape[0]):
            support = np.where(hz[row_idx])[0]
            if len(support) > 0:
                cx = np.mean([data_coords[q][0] for q in support])
                cy = np.mean([data_coords[q][1] for q in support])
                z_stab_coords_list.append((float(cx), float(cy)))
            else:
                z_stab_coords_list.append((0.0, 0.0))
        meta["z_stab_coords"] = z_stab_coords_list

        # Schedules — QLDPC codes generally use BP+OSD decoding, no plaquette schedule
        meta["x_schedule"] = None
        meta["z_schedule"] = None
        meta["stabiliser_schedule"] = {
            "x_rounds": {i: 0 for i in range(hx.shape[0])},
            "z_rounds": {i: 0 for i in range(hz.shape[0])},
            "n_rounds": 1,
            "description": (
                "Fully parallel: all stabilisers in round 0.  "
                "Decoding via BP+OSD rather than plaquette scheduling."
            ),
        }

        # ═══════════════════════════════════════════════════════════════════
        # LITERATURE / PROVENANCE
        # ═══════════════════════════════════════════════════════════════════
        meta["error_correction_zoo_url"] = "https://errorcorrectionzoo.org/c/qcga"
        meta["wikipedia_url"] = None
        meta["canonical_references"] = [
            "Bravyi, Cross, Gambetta, Maslov, Rall & Yoder, Nature 627, 778-782 (2024). arXiv:2308.07915",
            "Panteleev & Kalachev, Quantum 5, 585 (2021). arXiv:1904.02703",
        ]
        meta["connections"] = [
            "QLDPC code with constant row weight",
            "Circulant construction over Z_l x Z_m ensures CSS condition",
            "Generalised bicycle codes are the m=1 special case",
            "Efficient BP+OSD decoding",
        ]

        # ── Validate CSS structure ─────────────────────────────────
        validate_css_code(hx, hz, f"BB_{l}x{m}", raise_on_error=True)
        
        super().__init__(hx=hx, hz=hz, logical_x=logical_x, logical_z=logical_z, metadata=meta)
        
        # Store for qubit_coords and properties
        self._l = l
        self._m = m
        self._k = k

    # ─── Properties ────────────────────────────────────────────────
    @property
    def name(self) -> str:
        """Human-readable name, e.g. ``'BB_12x6'``."""
        return f"BB_{self._l}x{self._m}"

    @property
    def distance(self) -> int:
        """Code distance (looked up or computed; stored in metadata)."""
        d = self._metadata.get("distance")
        return d if d is not None else 1
    
    def qubit_coords(self) -> List[Tuple[float, float]]:
        """Return 2D qubit coordinates using torus grid layout.
        
        The BB code has two blocks of l×m qubits. We layout:
        - Block 0: positions (i, j) for i in [0,l), j in [0,m)
        - Block 1: positions (i + l + 1, j) for offset separation
        """
        coords = []
        block_size = self._l * self._m
        
        # Block 0: (i, j) grid
        for idx in range(block_size):
            i, j = idx // self._m, idx % self._m
            coords.append((float(i), float(j)))
        
        # Block 1: offset by (l + 1, 0) for visual separation
        for idx in range(block_size):
            i, j = idx // self._m, idx % self._m
            coords.append((float(i + self._l + 1), float(j)))
        
        return coords


# Pre-built BB codes from literature

def create_bb_gross_code() -> BivariateBicycleCode:
    """Create the [[144, 12, 12]] Gross code — a celebrated BB code.

    Parameters: ``l = 12``, ``m = 6``.

    .. math::

       A(x, y) = 1 + x^3 + x^6 + x^9, \\quad
       B(x, y) = y + y^2 + x^4 y + x^8 y^2

    Returns
    -------
    BivariateBicycleCode
        Code instance with ``n = 144``, ``k = 12``, ``d = 12``.
    """
    l, m = 12, 6
    A_terms = [(0, 0), (3, 0), (6, 0), (9, 0)]
    B_terms = [(0, 1), (0, 2), (4, 1), (8, 2)]
    
    code = BivariateBicycleCode(l, m, A_terms, B_terms)
    code._metadata["name"] = "Gross_144_12_12"
    code._metadata["distance"] = 12
    return code


def create_bb_small_12() -> BivariateBicycleCode:
    """Create a small [[72, 8, ?]] BB code.
    
    Good for testing: moderate size with reasonable parameters.
    Uses pure x-powers in A and pure y-powers in B for guaranteed k>0.
    """
    l, m = 6, 6
    # A = x + x^2 + x^3 (pure x-powers)
    A_terms = [(1, 0), (2, 0), (3, 0)]
    # B = y + y^2 + y^3 (pure y-powers)
    B_terms = [(0, 1), (0, 2), (0, 3)]
    
    code = BivariateBicycleCode(l, m, A_terms, B_terms)
    code._metadata["name"] = "BB_72_8"
    return code


def create_bb_tiny() -> BivariateBicycleCode:
    """Create a tiny BB code for fast testing.
    
    Parameters: l=3, m=3 giving n=18 qubits.
    """
    l, m = 3, 3
    A_terms = [(0, 0), (1, 0)]
    B_terms = [(0, 1), (1, 1)]
    
    code = BivariateBicycleCode(l, m, A_terms, B_terms)
    code._metadata["name"] = "BB_tiny_18"
    return code


# Convenience instances  
BBGrossCode = create_bb_gross_code
BBCode72 = create_bb_small_12
BBCodeTiny = create_bb_tiny
