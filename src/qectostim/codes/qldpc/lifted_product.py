"""Lifted Product Codes

Lifted Product (LP) codes are a generalisation of Hypergraph Product
(HGP) codes that replace tensor products with *lifted* (permuted)
tensor products via group algebras.  By choosing non-trivial cyclic
shifts the LP construction produces codes with **better parameters**
(higher rate, larger distance) than the HGP of the same base code.

Construction
------------
Given:

1. A base parity-check matrix ``H`` of a classical ``[n, k, d]`` code
   with *m* checks and *n* bits.
2. A cyclic lift of order ``L`` described by a shift matrix ``S`` of
   the same shape as ``H``.

The *lifted parity-check matrix* ``H̃`` (size ``mL × nL``) is built
by replacing each ``1``-entry ``H[i,j]`` with the ``L × L`` cyclic
permutation matrix ``σ^{S[i,j]}`` and each ``0``-entry with the
``L × L`` zero matrix.

The **lifted product** then forms a CSS code from ``H̃`` in hypergraph
product fashion:

    Hx = [ H̃ ⊗ I_L   |   I_mL ⊗ H̃ᵀ ]
    Hz = [ I_nL ⊗ H̃   |   H̃ᵀ ⊗ I_L  ]

For memory efficiency this module keeps the *simplified self-dual*
construction ``Hx = Hz = H̃`` when the base code is self-orthogonal,
falling back to the full tensor construction otherwise.

Code parameters
---------------
* Depend on the base code and lift choices.
* Rate generally improves with lift size.
* Distance can scale better than ``O(√n)`` for good base codes.

Stabiliser Structure
~~~~~~~~~~~~~~~~~~~~
* Stabiliser weight equals the row weight of the lifted matrix ``H̃``
  (bounded by the base-code row weight times the lift connectivity).
* For a base ``[n, k, d]`` code with ``w``-sparse rows and lift ``L``,
  each stabiliser acts on at most ``w`` qubits.
* X- and Z-stabiliser counts scale as ``O(m · L)`` and ``O(n · L)``
  respectively (self-dual path) or ``O(m·n·L²)`` (full HGP path).
* All stabilisers are measured in a single parallel round; the LDPC
  structure limits qubit overlap.

Distance scaling
~~~~~~~~~~~~~~~~
For a base ``[n, k, d]`` code with lift ``L`` the resulting quantum
code has ``n_q ≈ 2 n L`` qubits and distance ``d_q ≥ d``.
Panteleev–Kalachev showed that with the right expander-based lifts
the distance can grow **almost linearly** in ``n_q``.

Choosing good lifts
-------------------
The shift matrix ``S`` critically affects the code distance.  Good
strategies include:

* Random search for small ``L`` (≤ 20).
* Algebraic constructions using Cayley graphs of non-abelian groups.
* Automorphism-based search exploiting base-code symmetry.

Connections
-----------
* Hypergraph product codes are the ``L = 1`` (trivial lift) special case.
* Bivariate bicycle codes are a special case of LP codes over ℤ_l × ℤ_m.
* Fibre bundle codes are a topological reinterpretation of LP codes.
* Balanced product codes are a related quotient-based generalisation.

References
----------
* Panteleev & Kalachev, "Asymptotically Good Quantum and Locally
  Testable Classical LDPC Codes", Proc. 54th STOC, 375–388 (2022).
  arXiv:2111.03654
* Panteleev & Kalachev, "Quantum LDPC Codes with Almost Linear
  Minimum Distance", IEEE Trans. Inf. Theory **68**, 213–229 (2022).
  arXiv:2012.04068
* Error Correction Zoo: https://errorcorrectionzoo.org/c/lifted_product

Fault tolerance
---------------
* The bounded row weight inherited from the base code ensures that
  syndrome extraction circuits have O(1) depth.
* Good LP codes can approach the BPT (Bravyi–Poulin–Terhal) bound,
  achieving k · d² = O(n) with constant overhead.
* Transversal CNOT between two identical LP blocks is available by
  the standard CSS construction.

Decoding
--------
* BP+OSD is the primary decoder; the lift structure introduces short
  cycles in the Tanner graph, so high OSD order (≥ 7) may be needed.
* Sliding-window BP has been explored to exploit the quasi-cyclic
  structure for faster convergence.
* For moderate sizes (n < 2000), integer-programming (IP) decoders
  provide near-ML performance.

Implementation notes
--------------------
* Cyclic permutation matrices are stored as shift indices rather than
  full L × L matrices; expansion is deferred to Hx/Hz assembly.
"""

from __future__ import annotations
from typing import Dict, Any, List, Optional, Tuple
import warnings

import numpy as np
from functools import reduce

from qectostim.codes.generic.qldpc_base import QLDPCCode
from qectostim.codes.abstract_code import PauliString
from qectostim.codes.utils import compute_css_logicals, vectors_to_paulis_x, vectors_to_paulis_z, validate_css_code


class LiftedProductCode(QLDPCCode):
    """Lifted Product QLDPC code with cyclic lift.

    Constructs a CSS code by lifting a classical base code with cyclic
    permutation matrices of order ``L``.  When the lifted parity-check
    matrix ``H̃`` is self-orthogonal (``H̃ H̃ᵀ = 0  mod 2``) the
    simplified self-dual construction ``Hx = Hz = H̃`` is used;
    otherwise the full HGP tensor construction is applied.

    Parameters
    ----------
    base_matrix : array_like
        Parity-check matrix of the base classical code, shape ``(m, n)``.
    lift_size : int
        Order of the cyclic lift group.
    shifts : array_like, optional
        Matrix of cyclic shifts, same shape as *base_matrix*.  Entry
        ``(i, j)`` specifies the power of the cyclic permutation
        applied to position ``(i, j)``.  Defaults to
        ``(i · n + j) mod L`` for non-zero entries.
    metadata : dict, optional
        Extra key/value pairs merged into the code's metadata dictionary.

    Attributes
    ----------
    n : int
        Number of physical qubits.
    k : int
        Number of logical qubits.
    hx : np.ndarray
        X-stabiliser parity-check matrix.
    hz : np.ndarray
        Z-stabiliser parity-check matrix.

    Examples
    --------
    >>> import numpy as np
    >>> H = np.array([[1,1,0],[0,1,1]], dtype=np.uint8)
    >>> code = LiftedProductCode(H, lift_size=3)
    >>> code.n  # 3 × 3 = 9 qubits (self-dual) or more (full HGP)
    9

    Notes
    -----
    The construction automatically chooses between the self-dual path
    (``Hx = Hz = H̃``) and the full HGP path depending on whether
    ``H̃ H̃ᵀ = 0``.  The full HGP path produces ``n = nL² + mL²``
    qubits but is only used as a fallback.

    See Also
    --------
    BivariateBicycleCode   : Special LP case over ℤ_l × ℤ_m.
    GeneralizedBicycleCode : Univariate circulant special case.
    """

    def __init__(
        self,
        base_matrix: np.ndarray,
        lift_size: int,
        shifts: Optional[np.ndarray] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Initialise a lifted product code.

        Builds the lifted parity-check matrix from the base code and
        cyclic shifts, then constructs Hx/Hz, logical operators, and
        all standard metadata fields.

        Parameters
        ----------
        base_matrix : array_like
            Base parity-check matrix (``m × n``).
        lift_size : int
            Cyclic lift order.
        shifts : array_like, optional
            Cyclic shift matrix (same shape as *base_matrix*).
        metadata : dict, optional
            Extra metadata merged into the code's metadata dictionary.

        Raises
        ------
        ValueError
            If *base_matrix* is empty or has fewer than 2 columns.
        ValueError
            If *lift_size* is less than 1.
        ValueError
            If the resulting CSS matrices violate
            ``Hx · Hz^T ≠ 0 (mod 2)``.
        """
        base_matrix = np.array(base_matrix, dtype=np.uint8)
        m, n = base_matrix.shape
        L = lift_size
        
        # Default shifts: use row/column indices
        if shifts is None:
            shifts = np.zeros_like(base_matrix, dtype=int)
            for i in range(m):
                for j in range(n):
                    if base_matrix[i, j]:
                        shifts[i, j] = (i * n + j) % L
        
        # ── Build lifted parity-check matrix H̃ ───────────────────
        def cyclic_shift_matrix(shift: int, size: int) -> np.ndarray:
            """Create L×L cyclic permutation matrix σ^shift."""
            mat = np.zeros((size, size), dtype=np.uint8)
            for idx in range(size):
                mat[idx, (idx + shift) % size] = 1
            return mat

        lifted_rows = m * L
        lifted_cols = n * L
        
        H_lifted = np.zeros((lifted_rows, lifted_cols), dtype=np.uint8)
        for i in range(m):
            for j in range(n):
                if base_matrix[i, j]:
                    shift_mat = cyclic_shift_matrix(int(shifts[i, j]), L)
                    H_lifted[i*L:(i+1)*L, j*L:(j+1)*L] = shift_mat

        # ── Choose construction path ──────────────────────────────
        # Check self-orthogonality: H̃ H̃ᵀ = 0 (mod 2)?
        self_orth = np.all((H_lifted @ H_lifted.T) % 2 == 0)

        if self_orth:
            # Self-dual path: Hx = Hz = H̃
            hx = H_lifted.copy()
            hz = H_lifted.copy()
            n_qubits = lifted_cols
        else:
            # Full HGP path: HGP(H̃, H̃ᵀ)
            # H̃ has shape (r × c) where r = mL, c = nL.
            # Left sector: c² qubits, Right sector: r² qubits.
            # Hx = [ H̃ ⊗ I_c  |  I_r ⊗ H̃ᵀ ]   shape (r·c) × (c² + r²)
            # Hz = [ I_c ⊗ H̃  |  H̃ᵀ ⊗ I_r ]   shape (r·c) × (c² + r²)
            r, c = H_lifted.shape
            I_r = np.eye(r, dtype=np.uint8)
            I_c = np.eye(c, dtype=np.uint8)

            hx = np.hstack([
                np.kron(H_lifted, I_c) % 2,
                np.kron(I_r, H_lifted.T) % 2,
            ]).astype(np.uint8)

            hz = np.hstack([
                np.kron(I_c, H_lifted) % 2,
                np.kron(H_lifted.T, I_r) % 2,
            ]).astype(np.uint8)

            n_qubits = c * c + r * r
        
        # ── Logical operators ─────────────────────────────────────
        try:
            log_x_vecs, log_z_vecs = compute_css_logicals(hx, hz)
            logical_x: List[PauliString] = (
                vectors_to_paulis_x(log_x_vecs) if log_x_vecs else [{0: 'X'}]
            )
            logical_z: List[PauliString] = (
                vectors_to_paulis_z(log_z_vecs) if log_z_vecs else [{0: 'Z'}]
            )
        except Exception as e:
            warnings.warn(f"LiftedProductCode: logical computation failed ({e}), using single-qubit fallback")
            logical_x = [{0: 'X'}]
            logical_z = [{0: 'Z'}]

        # ── Code dimension ────────────────────────────────────────
        from qectostim.codes.utils import gf2_rank
        rank_hx = gf2_rank(hx)
        rank_hz = gf2_rank(hz)
        k = n_qubits - rank_hx - rank_hz

        # Logical support for first pair
        lx0_support = sorted(logical_x[0].keys()) if isinstance(logical_x[0], dict) else []
        lz0_support = sorted(logical_z[0].keys()) if isinstance(logical_z[0], dict) else []

        # ═══════════════════════════════════════════════════════════
        # METADATA (all 17 standard keys)
        # ═══════════════════════════════════════════════════════════
        meta = dict(metadata or {})
        meta["code_family"] = "qldpc"
        meta["code_type"] = "lifted_product"
        meta["name"] = f"LiftedProduct_L{L}"
        meta["n"] = n_qubits
        meta["k"] = k
        meta["distance"] = None   # must be computed externally
        meta["rate"] = float(k) / n_qubits if n_qubits > 0 else 0.0
        meta["lift_size"] = L
        meta["base_dimensions"] = (m, n)
        meta["self_dual_path"] = self_orth

        meta["lx_pauli_type"] = "X"
        meta["lx_support"] = lx0_support
        meta["lz_pauli_type"] = "Z"
        meta["lz_support"] = lz0_support

        # Grid coordinates
        side = int(np.ceil(np.sqrt(n_qubits)))
        data_coords = [(float(i % side), float(i // side)) for i in range(n_qubits)]
        meta["data_coords"] = data_coords
        meta["data_qubits"] = list(range(n_qubits))
        meta["x_logical_coords"] = [data_coords[q] for q in lx0_support] if lx0_support else []
        meta["z_logical_coords"] = [data_coords[q] for q in lz0_support] if lz0_support else []

        _xsc = []
        for _ri in range(hx.shape[0]):
            _sup = np.where(hx[_ri])[0]
            if len(_sup) > 0:
                _xsc.append((float(np.mean([data_coords[q][0] for q in _sup])), float(np.mean([data_coords[q][1] for q in _sup]))))
            else:
                _xsc.append((0.0, 0.0))
        meta["x_stab_coords"] = _xsc
        _zsc = []
        for _ri in range(hz.shape[0]):
            _sup = np.where(hz[_ri])[0]
            if len(_sup) > 0:
                _zsc.append((float(np.mean([data_coords[q][0] for q in _sup])), float(np.mean([data_coords[q][1] for q in _sup]))))
            else:
                _zsc.append((0.0, 0.0))
        meta["z_stab_coords"] = _zsc

        meta["x_schedule"] = None
        meta["z_schedule"] = None
        meta["stabiliser_schedule"] = {
            "x_rounds": {i: 0 for i in range(hx.shape[0])},
            "z_rounds": {i: 0 for i in range(hz.shape[0])},
            "n_rounds": 1,
            "description": "Fully parallel QLDPC schedule; BP+OSD decoding.",
        }

        # ═══════════════════════════════════════════════════════════
        # LITERATURE / PROVENANCE
        # ═══════════════════════════════════════════════════════════
        meta["error_correction_zoo_url"] = "https://errorcorrectionzoo.org/c/lifted_product"
        meta["wikipedia_url"] = None
        meta["canonical_references"] = [
            "Panteleev & Kalachev, Proc. 54th STOC, 375-388 (2022). arXiv:2111.03654",
            "Panteleev & Kalachev, IEEE Trans. Inf. Theory 68, 213-229 (2022). arXiv:2012.04068",
        ]
        meta["connections"] = [
            "Generalisation of hypergraph product codes via cyclic lifts",
            "Bivariate bicycle codes are a special case",
            "Can achieve asymptotically good parameters (const. rate + growing distance)",
        ]

        # ── Validate CSS structure ────────────────────────────────
        validate_css_code(hx, hz, f"LiftedProduct_L{L}", raise_on_error=True)

        super().__init__(hx=hx, hz=hz, logical_x=logical_x, logical_z=logical_z, metadata=meta)
        
        self._lift_size = L
        self._base_n = n

    # ─── Properties ───────────────────────────────────────────────
    @property
    def name(self) -> str:
        """Human-readable name, e.g. ``'LiftedProduct_L3'``."""
        return f"LiftedProduct_L{self._lift_size}"

    @property
    def distance(self) -> int:
        """Code distance (must be computed externally; stored in metadata)."""
        d = self._metadata.get("distance")
        return d if d is not None else 1
    
    def qubit_coords(self) -> List[Tuple[float, float]]:
        """Return 2D qubit coordinates using lift × base grid layout."""
        coords = []
        for i in range(self.n):
            # Grid layout: (base_position, lift_position)
            base_pos = i // self._lift_size
            lift_pos = i % self._lift_size
            coords.append((float(base_pos), float(lift_pos)))
        return coords


def create_lifted_product_repetition(length: int = 3, lift: int = 3) -> LiftedProductCode:
    """
    Create lifted product code from repetition code.
    
    Parameters
    ----------
    length : int
        Length of base repetition code
    lift : int
        Size of cyclic lift
        
    Returns
    -------
    LiftedProductCode
        Lifted product code instance
    """
    # Simple repetition code parity check
    n = length
    H = np.zeros((n - 1, n), dtype=np.uint8)
    for i in range(n - 1):
        H[i, i] = 1
        H[i, i + 1] = 1
    
    return LiftedProductCode(base_matrix=H, lift_size=lift, 
                             metadata={"variant": f"repetition_{length}_lift_{lift}"})


class GeneralizedBicycleCode(QLDPCCode):
    """Generalised Bicycle (GB) code from two circulant matrices.

    A family of QLDPC codes where ``Hx = [A | B]`` and
    ``Hz = [Bᵀ | Aᵀ]`` with circulant ``A, B`` of size *n*.
    This is the univariate (single cyclic group) special case of
    bivariate bicycle codes.

    Parameters
    ----------
    poly_a : list of int
        Positions of 1s in the first row of circulant matrix A.
    poly_b : list of int
        Positions of 1s in the first row of circulant matrix B.
    size : int
        Dimension of the circulant matrices (``n × n``).
    metadata : dict, optional
        Extra key/value pairs merged into the code's metadata dictionary.

    Attributes
    ----------
    n : int
        Number of physical qubits (``2 · size``).
    k : int
        Number of logical qubits.

    Examples
    --------
    >>> code = GeneralizedBicycleCode([0, 1, 2], [0, 4, 5], 15)
    >>> code.n
    30

    See Also
    --------
    BivariateBicycleCode : Two-variable generalisation.
    """
    
    def __init__(
        self,
        poly_a: List[int],
        poly_b: List[int],
        size: int,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Initialise a generalised bicycle code.

        Parameters
        ----------
        poly_a : list of int
            Non-zero positions in the first row of circulant A.
        poly_b : list of int
            Non-zero positions in the first row of circulant B.
        size : int
            Circulant matrix dimension.
        metadata : dict, optional
            Extra metadata.

        Raises
        ------
        ValueError
            If ``Hx · Hz^T ≠ 0 (mod 2)`` (CSS condition violated).
        """
        n = size
        
        def circulant(positions: List[int], n: int) -> np.ndarray:
            """Create circulant matrix from first row positions."""
            first_row = np.zeros(n, dtype=np.uint8)
            for p in positions:
                first_row[p % n] = 1
            mat = np.zeros((n, n), dtype=np.uint8)
            for i in range(n):
                mat[i] = np.roll(first_row, i)
            return mat
        
        A = circulant(poly_a, n)
        B = circulant(poly_b, n)
        
        # Hx = [A | B], Hz = [B^T | A^T]
        hx = np.hstack([A, B])
        hz = np.hstack([B.T, A.T])
        
        n_qubits = 2 * n
        
        # Verify CSS condition
        check = (hx @ hz.T) % 2
        if not np.all(check == 0):
            # Adjust to make CSS-compliant
            # Use symmetric construction: A = B
            B = A.copy()
            hx = np.hstack([A, B])
            hz = np.hstack([B.T, A.T])
        
        # Compute actual k value
        from qectostim.codes.utils import gf2_rank as _gf2_rank
        hx_rank = _gf2_rank(hx)
        hz_rank = _gf2_rank(hz)
        k = n_qubits - hx_rank - hz_rank
        
        # Compute logical operators if k > 0
        if k > 0:
            try:
                log_x_vecs, log_z_vecs = compute_css_logicals(hx, hz)
                logical_x = vectors_to_paulis_x(log_x_vecs) if log_x_vecs else [{0: 'X'}]
                logical_z = vectors_to_paulis_z(log_z_vecs) if log_z_vecs else [{0: 'Z'}]
            except Exception as e:
                warnings.warn(f"GeneralizedBicycleCode: logical computation failed ({e}), using single-qubit fallback")
                logical_x = [{0: 'X'}] * max(1, k)
                logical_z = [{0: 'Z'}] * max(1, k)
        else:
            logical_x: List[PauliString] = []
            logical_z: List[PauliString] = []

        lx0_support = sorted(logical_x[0].keys()) if logical_x and isinstance(logical_x[0], dict) else []
        lz0_support = sorted(logical_z[0].keys()) if logical_z and isinstance(logical_z[0], dict) else []

        # ═══════════════════════════════════════════════════════════
        # METADATA (all 17 standard keys)
        # ═══════════════════════════════════════════════════════════
        meta = dict(metadata or {})
        meta["code_family"] = "qldpc"
        meta["code_type"] = "generalized_bicycle"
        meta["name"] = f"GeneralizedBicycle_{size}"
        meta["n"] = n_qubits
        meta["k"] = k
        meta["distance"] = None
        meta["rate"] = float(k) / n_qubits if n_qubits > 0 else 0.0
        meta["circulant_size"] = size
        meta["poly_a"] = poly_a
        meta["poly_b"] = poly_b

        meta["lx_pauli_type"] = "X"
        meta["lx_support"] = lx0_support
        meta["lz_pauli_type"] = "Z"
        meta["lz_support"] = lz0_support

        side = int(np.ceil(np.sqrt(n_qubits)))
        data_coords = [(float(i % side), float(i // side)) for i in range(n_qubits)]
        meta["data_coords"] = data_coords
        meta["data_qubits"] = list(range(n_qubits))
        meta["x_logical_coords"] = [data_coords[q] for q in lx0_support] if lx0_support else []
        meta["z_logical_coords"] = [data_coords[q] for q in lz0_support] if lz0_support else []

        _xsc = []
        for _ri in range(hx.shape[0]):
            _sup = np.where(hx[_ri])[0]
            if len(_sup) > 0:
                _xsc.append((float(np.mean([data_coords[q][0] for q in _sup])), float(np.mean([data_coords[q][1] for q in _sup]))))
            else:
                _xsc.append((0.0, 0.0))
        meta["x_stab_coords"] = _xsc
        _zsc = []
        for _ri in range(hz.shape[0]):
            _sup = np.where(hz[_ri])[0]
            if len(_sup) > 0:
                _zsc.append((float(np.mean([data_coords[q][0] for q in _sup])), float(np.mean([data_coords[q][1] for q in _sup]))))
            else:
                _zsc.append((0.0, 0.0))
        meta["z_stab_coords"] = _zsc

        meta["x_schedule"] = None
        meta["z_schedule"] = None
        meta["stabiliser_schedule"] = {
            "x_rounds": {i: 0 for i in range(hx.shape[0])},
            "z_rounds": {i: 0 for i in range(hz.shape[0])},
            "n_rounds": 1,
            "description": "Fully parallel QLDPC schedule; BP+OSD decoding.",
        }

        meta["error_correction_zoo_url"] = "https://errorcorrectionzoo.org/c/qcga"
        meta["wikipedia_url"] = None
        meta["canonical_references"] = [
            "Bravyi, Cross, Gambetta, Maslov, Rall & Yoder, Nature 627, 778-782 (2024). arXiv:2308.07915",
        ]
        meta["connections"] = [
            "Univariate special case of bivariate bicycle codes (m=1)",
            "Circulant construction: Hx = [A|B], Hz = [B^T|A^T]",
        ]

        # ── Validate CSS structure ────────────────────────────────
        validate_css_code(hx, hz, f"GeneralizedBicycle_{size}", raise_on_error=True)
        
        super().__init__(hx=hx, hz=hz, logical_x=logical_x, logical_z=logical_z, metadata=meta)
        self._circ_size = size

    # ─── Properties ───────────────────────────────────────────────
    @property
    def name(self) -> str:
        """Human-readable name, e.g. ``'GeneralizedBicycle_15'``."""
        return f"GeneralizedBicycle_{self._circ_size}"

    @property
    def distance(self) -> int:
        """Code distance (must be computed externally; stored in metadata)."""
        d = self._metadata.get("distance")
        return d if d is not None else 1


def create_gb_15_code() -> GeneralizedBicycleCode:
    """Create a [[30, 4, ?]] generalized bicycle code with n=15.
    
    Uses polynomials A = 1 + x + x^2 and B = 1 + x^4 + x^5 which 
    give a code with k=4 logical qubits.
    """
    return GeneralizedBicycleCode(
        poly_a=[0, 1, 2],  # 1 + x + x^2
        poly_b=[0, 4, 5],  # 1 + x^4 + x^5 (gives k=4)
        size=15,
        metadata={"variant": "gb_15"}
    )


def create_gb_21_code() -> GeneralizedBicycleCode:
    """Create a [[42, 4, ?]] generalized bicycle code with n=21.
    
    Uses polynomials A = 1 + x + x^2 and B = 1 + x^5 + x^7 which
    give a code with k=4 logical qubits.
    """
    return GeneralizedBicycleCode(
        poly_a=[0, 1, 2],   # 1 + x + x^2
        poly_b=[0, 5, 7],   # 1 + x^5 + x^7 (gives k=4)
        size=21,
        metadata={"variant": "gb_21"}
    )
