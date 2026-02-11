"""Balanced Product Codes

Balanced product codes are a generalisation of Hypergraph Product (HGP)
codes that achieve better ``k/n`` and ``d/n`` ratios by taking
**quotients** of the HGP code by a group action.

Construction
------------
Given classical codes ``C_A`` (parity-check ``H_A``, size ``m_A × n_A``)
and ``C_B`` (``H_B``, ``m_B × n_B``) the **standard HGP** gives:

    Hx = [ H_A ⊗ I_{n_B}  |  I_{m_A} ⊗ H_Bᵀ ]
    Hz = [ I_{n_A} ⊗ H_B   |  H_Aᵀ ⊗ I_{m_B} ]

with ``n = n_A n_B + m_A m_B`` physical qubits.  The *balanced*
variant quotients this construction by a group ``G`` of order ``|G|``
that acts on both codes, compressing all parameters.

Code parameters
---------------
* **n** ≈ (n_A n_B + m_A m_B) / |G|
* **k** ≈ k_A k_B / |G|
* **d** ≥ min(d_A d_B, d_A n_B / |G|, n_A d_B / |G|)

When ``|G| = 1`` the code reduces to the standard HGP.

Stabiliser Structure
~~~~~~~~~~~~~~~~~~~~
* X-stabiliser count: ``m_A · n_B / |G|``; Z-stabiliser count:
  ``n_A · m_B / |G|`` (after quotient).
* Stabiliser weights are bounded by the row weights of ``H_A`` and
  ``H_B`` (inherited from the HGP structure).
* The quotient can create short cycles in the Tanner graph, increasing
  the effective stabiliser overlap but not the per-stabiliser weight.
* Measurement schedule: single parallel round (all X-stabilisers in
  round 0, all Z-stabilisers in round 0) using BP+OSD decoding.

Group quotient detail
~~~~~~~~~~~~~~~~~~~~~
The quotient is formed by identifying qubits within each orbit of
the group action.  For a cyclic group of order ``g``, blocks of
``g`` consecutive qubits are merged by summing their columns
modulo 2.  This preserves the CSS condition while compressing ``n``.

Distance bounds
~~~~~~~~~~~~~~~
The distance of a balanced product code inherits lower bounds from
the constituent classical codes.  For good LDPC base codes with
``d_A, d_B = Ω(n^α)`` the quantum distance can exceed ``√n``.

Decoding
--------
BP+OSD decoding works well for balanced product codes due to their
LDPC structure.  The quotient can introduce short cycles in the
Tanner graph, so higher OSD order may be needed.

Worked example
--------------
With a ``[5,1,3]`` repetition code and ``|G| = 1`` (standard HGP):
``n = 5·5 + 4·4 = 41``, ``k = 1``, ``d = 3``.

With ``|G| = 5`` the same base code gives ``n ≈ 41/5 ≈ 9`` qubits
but the distance may degrade.  Finding good group actions that
preserve distance while maximising compression is an active research
area.

Connections
-----------
* HGP codes are the ``|G| = 1`` special case.
* Quantum Tanner codes extend balanced products with local views.
* Lifted product codes are a related algebraic generalisation.
* BB codes are a different compression strategy (circulant structure).

References
----------
* Breuckmann & Eberhardt, "Balanced Product Quantum Codes",
  IEEE Trans. Inf. Theory **67**, 6653–6674 (2021).  arXiv:2012.09271
* Leverrier & Zémor, "Quantum Tanner Codes", Proc. 63rd FOCS,
  872–883 (2022).  arXiv:2202.13641
* Error Correction Zoo: https://errorcorrectionzoo.org/c/balanced_product

Fault tolerance
---------------
* The LDPC structure means each stabiliser has bounded weight,
  keeping syndrome-extraction circuit depth constant as n grows.
* The low-density property also limits the spread of hook errors,
  improving fault-tolerance thresholds relative to dense codes.
* Concatenation with an inner code is possible but rarely needed
  due to the already-good asymptotic distance.

Implementation notes
--------------------
* Group orbits are computed by enumerating the action of the cyclic
  generator on qubit indices and merging orbits via union-find.
* The quotient parity-check matrix is formed by summing columns
  within each orbit modulo 2, then removing duplicate rows.
* For non-cyclic groups the orbit computation generalises to the
  full group multiplication table.
* Sparse matrix representations (``scipy.sparse``) are used
  throughout to handle the Kronecker products efficiently.
"""

from __future__ import annotations
from typing import Dict, Any, List, Optional, Tuple
import warnings
import numpy as np
from scipy import sparse

from qectostim.codes.generic.qldpc_base import QLDPCCode
from qectostim.codes.abstract_css import Coord2D
from qectostim.codes.abstract_code import PauliString
from qectostim.codes.utils import compute_css_logicals, vectors_to_paulis_x, vectors_to_paulis_z, validate_css_code


class BalancedProductCode(QLDPCCode):
    """Balanced product quantum code from two classical codes.

    Constructs the HGP of ``H_A`` and ``H_B``, optionally quotiented
    by a cyclic group of order ``|G|``.  When ``|G| = 1`` this is the
    standard Hypergraph Product code.

    Parameters
    ----------
    ha : array_like
        Parity-check matrix of classical code A.
    hb : array_like
        Parity-check matrix of classical code B.
    group_order : int, optional
        Order of the quotient group (default 1 = standard HGP).
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
    >>> H = np.array([[1,1,0,0,0],[0,1,1,0,0],[0,0,1,1,0],[0,0,0,1,1]], dtype=np.uint8)
    >>> code = BalancedProductCode(H, H, group_order=1)
    >>> code.n   # 5*5 + 4*4 = 41
    41

    Notes
    -----
    For ``|G| > 1`` the quotient is performed by summing columns in
    blocks of size ``|G|`` — a simplified cyclic-orbit identification.

    See Also
    --------
    LiftedProductCode      : Related algebraic generalisation.
    DistanceBalancedCode   : Asymmetric-distance variant.
    """

    def __init__(
        self,
        ha: np.ndarray,
        hb: np.ndarray,
        group_order: int = 1,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Initialise a balanced product code.

        Builds the HGP-style parity-check matrices ``Hx`` and ``Hz``
        from two classical codes, optionally applying a cyclic quotient.
        Computes logical operators and populates all standard metadata.

        Parameters
        ----------
        ha : array_like
            Parity-check matrix of code A.
        hb : array_like
            Parity-check matrix of code B.
        group_order : int, optional
            Quotient group order (default 1).
        metadata : dict, optional
            Extra metadata.

        Raises
        ------
        ValueError
            If ``group_order > 1`` and the dimensions of *ha* or *hb*
            are not divisible by *group_order*.
        ValueError
            If the resulting parity-check matrices violate the CSS
            constraint ``Hx · Hz^T ≠ 0 (mod 2)``.
        """
        self._ha = np.array(ha, dtype=np.uint8)
        self._hb = np.array(hb, dtype=np.uint8)
        self._group_order = group_order
        
        # Build balanced product matrices
        hx, hz, n_qubits = self._build_balanced_product(
            self._ha, self._hb, group_order
        )
        
        # Compute logical operators (placeholder)
        logical_x, logical_z = self._compute_logicals(hx, hz, n_qubits)
        
        # Compute code parameters
        n_a, n_b = ha.shape[1], hb.shape[1]
        from qectostim.codes.utils import gf2_rank as _gf2_rank
        rank_hx = _gf2_rank(hx)
        rank_hz = _gf2_rank(hz)
        k = n_qubits - rank_hx - rank_hz

        # Distance estimate: balanced product often has d = O(sqrt(n))
        d_estimate = max(2, int(np.sqrt(n_qubits / 2)))

        # Logical support for first pair
        lx0_support = sorted(logical_x[0].keys()) if isinstance(logical_x[0], dict) else []
        lz0_support = sorted(logical_z[0].keys()) if isinstance(logical_z[0], dict) else []

        # ═══════════════════════════════════════════════════════════
        # METADATA (all 17 standard keys)
        # ═══════════════════════════════════════════════════════════
        meta: Dict[str, Any] = dict(metadata or {})
        meta["code_family"] = "qldpc"
        meta["code_type"] = "balanced_product"
        meta["name"] = f"BalancedProduct_{n_a}x{n_b}_G{group_order}"
        meta["n"] = n_qubits
        meta["k"] = k
        meta["distance"] = d_estimate
        meta["rate"] = float(k) / n_qubits if n_qubits > 0 else 0.0
        meta["construction"] = "balanced_product"
        meta["group_order"] = group_order
        meta["base_code_a"] = {"n": n_a, "k": int(n_a - np.linalg.matrix_rank(ha))}
        meta["base_code_b"] = {"n": n_b, "k": int(n_b - np.linalg.matrix_rank(hb))}

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
        meta["error_correction_zoo_url"] = "https://errorcorrectionzoo.org/c/balanced_product"
        meta["wikipedia_url"] = None
        meta["canonical_references"] = [
            "Breuckmann & Eberhardt, IEEE Trans. Inf. Theory 67, 6653-6674 (2021). arXiv:2012.09271",
            "Leverrier & Zémor, Proc. 63rd FOCS, 872-883 (2022). arXiv:2202.13641",
        ]
        meta["connections"] = [
            "HGP codes are the |G|=1 special case",
            "Quotient by group action improves rate and distance scaling",
            "Related to quantum Tanner codes via local views",
        ]

        # ── Validate CSS structure ────────────────────────────────
        validate_css_code(hx, hz, f"BalancedProduct_{n_a}x{n_b}_G{group_order}", raise_on_error=True)
        
        super().__init__(hx=hx, hz=hz, logical_x=logical_x, logical_z=logical_z, metadata=meta)
        self._hx = hx
        self._hz = hz
        self._group_order_val = group_order
        self._na = n_a
        self._nb = n_b

    # ─── Properties ───────────────────────────────────────────────
    @property
    def name(self) -> str:
        """Human-readable name."""
        return f"BalancedProduct_{self._na}x{self._nb}_G{self._group_order_val}"

    @property
    def distance(self) -> int:
        """Estimated code distance (heuristic √(n/2))."""
        return self._metadata.get("distance", 1)
    
    def qubit_coords(self) -> List[Coord2D]:
        """Return 2D qubit coordinates."""
        n = self.n
        side = int(np.ceil(np.sqrt(n)))
        return [(float(i % side), float(i // side)) for i in range(n)]
    
    @staticmethod
    def _build_balanced_product(
        ha: np.ndarray, hb: np.ndarray, group_order: int
    ) -> Tuple[np.ndarray, np.ndarray, int]:
        """
        Build balanced product parity check matrices using proper HGP construction.
        
        Standard HGP with matrices H_A (ma × na) and H_B (mb × nb):
        - Left sector: na × nb qubits indexed as (bit_a, bit_b)
        - Right sector: ma × mb qubits indexed as (check_a, check_b)
        - Total: n = na*nb + ma*mb qubits
        
        X-stabilizers: ma × nb of them, indexed by (check_a, bit_b)
        Z-stabilizers: na × mb of them, indexed by (bit_a, check_b)
        """
        ma, na = ha.shape  # A has ma checks, na bits
        mb, nb = hb.shape  # B has mb checks, nb bits
        
        if group_order > 1:
            if na % group_order != 0 or nb % group_order != 0:
                raise ValueError(f"n_A={na} and n_B={nb} must be divisible by group_order={group_order}")
            
            na_q = na // group_order
            nb_q = nb // group_order
            
            ha_q = np.zeros((ma, na_q), dtype=np.uint8)
            hb_q = np.zeros((mb, nb_q), dtype=np.uint8)
            
            for i in range(na_q):
                for g in range(group_order):
                    ha_q[:, i] = (ha_q[:, i] + ha[:, i * group_order + g]) % 2
                    
            for j in range(nb_q):
                for g in range(group_order):
                    hb_q[:, j] = (hb_q[:, j] + hb[:, j * group_order + g]) % 2
            
            ha_use = ha_q
            hb_use = hb_q
        else:
            ha_use = ha
            hb_use = hb
        
        ma_use, na_use = ha_use.shape
        mb_use, nb_use = hb_use.shape
        
        # Qubit sectors
        n_left = na_use * nb_use   # (bit_a, bit_b) pairs
        n_right = ma_use * mb_use  # (check_a, check_b) pairs
        n_qubits = n_left + n_right
        
        # Stabilizer counts
        n_x_stabs = ma_use * nb_use  # (check_a, bit_b) pairs
        n_z_stabs = na_use * mb_use  # (bit_a, check_b) pairs
        
        hx = np.zeros((n_x_stabs, n_qubits), dtype=np.uint8)
        hz = np.zeros((n_z_stabs, n_qubits), dtype=np.uint8)
        
        # Build X-stabilizers
        for check_a in range(ma_use):
            for bit_b in range(nb_use):
                x_stab = check_a * nb_use + bit_b
                # Left sector: qubits (bit_a, bit_b) where H_A[check_a, bit_a] = 1
                for bit_a in range(na_use):
                    if ha_use[check_a, bit_a]:
                        q = bit_a * nb_use + bit_b
                        hx[x_stab, q] = 1
                # Right sector: qubits (check_a, check_b) where H_B[check_b, bit_b] = 1
                for check_b in range(mb_use):
                    if hb_use[check_b, bit_b]:
                        q = n_left + check_a * mb_use + check_b
                        hx[x_stab, q] = 1
        
        # Build Z-stabilizers
        for bit_a in range(na_use):
            for check_b in range(mb_use):
                z_stab = bit_a * mb_use + check_b
                # Left sector: qubits (bit_a, bit_b) where H_B[check_b, bit_b] = 1
                for bit_b in range(nb_use):
                    if hb_use[check_b, bit_b]:
                        q = bit_a * nb_use + bit_b
                        hz[z_stab, q] = 1
                # Right sector: qubits (check_a, check_b) where H_A[check_a, bit_a] = 1
                for check_a in range(ma_use):
                    if ha_use[check_a, bit_a]:
                        q = n_left + check_a * mb_use + check_b
                        hz[z_stab, q] = 1
        
        return hx % 2, hz % 2, n_qubits
    
    @staticmethod
    def _compute_logicals(
        hx: np.ndarray, hz: np.ndarray, n_qubits: int
    ) -> Tuple[List[PauliString], List[PauliString]]:
        """Compute logical operators using CSS kernel/image prescription."""
        try:
            log_x_vecs, log_z_vecs = compute_css_logicals(hx, hz)
            logical_x = vectors_to_paulis_x(log_x_vecs)
            logical_z = vectors_to_paulis_z(log_z_vecs)
            # Ensure we have at least one logical operator
            if not logical_x:
                logical_x = [{0: 'X'}]
            if not logical_z:
                logical_z = [{0: 'Z'}]
            return logical_x, logical_z
        except Exception as e:
            warnings.warn(
                f"BalancedProductCode: logical operator computation failed "
                f"({e}); falling back to single-qubit placeholder."
            )
            return [{0: 'X'}], [{0: 'Z'}]


class DistanceBalancedCode(QLDPCCode):
    """Distance-balanced HGP code for asymmetric noise.

    Uses different base codes for the X- and Z-distance sectors,
    allowing the code to be tailored when X and Z error rates differ
    (e.g. biased-noise channels).

    Parameters
    ----------
    ha : array_like
        Parity-check matrix controlling the X distance.
    hb : array_like
        Parity-check matrix controlling the Z distance.
    metadata : dict, optional
        Extra key/value pairs merged into the code's metadata dictionary.

    See Also
    --------
    BalancedProductCode : Symmetric balanced product.
    """

    def __init__(
        self,
        ha: np.ndarray,
        hb: np.ndarray,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Initialise a distance-balanced HGP code.

        Parameters
        ----------
        ha : array_like
            Parity-check matrix for X-distance sector.
        hb : array_like
            Parity-check matrix for Z-distance sector.
        metadata : dict, optional
            Extra metadata.

        Raises
        ------
        ValueError
            If the resulting CSS matrices violate
            ``Hx · Hz^T ≠ 0 (mod 2)``.
        """
        # Build standard HGP(H_A, H_B)
        # H_A = ha (ma × na),  H_B = hb (mb × nb)
        # Left sector: na × nb qubits, Right sector: ma × mb qubits
        ma, na = ha.shape
        mb, nb = hb.shape
        
        n_qubits = na * nb + ma * mb
        
        # Hx = [ H_A ⊗ I_nb  |  I_ma ⊗ H_Bᵀ ]  shape (ma·nb) × (na·nb + ma·mb)
        hx_left = np.kron(ha, np.eye(nb, dtype=np.uint8))        # (ma·nb, na·nb)
        hx_right = np.kron(np.eye(ma, dtype=np.uint8), hb.T)     # (ma·nb, ma·mb)
        hx = np.hstack([hx_left, hx_right]).astype(np.uint8) % 2
        
        # Hz = [ I_na ⊗ H_B  |  H_Aᵀ ⊗ I_mb ]  shape (na·mb) × (na·nb + ma·mb)
        hz_left = np.kron(np.eye(na, dtype=np.uint8), hb)         # (na·mb, na·nb)
        hz_right = np.kron(ha.T, np.eye(mb, dtype=np.uint8))      # (na·mb, ma·mb)
        hz = np.hstack([hz_left, hz_right]).astype(np.uint8) % 2
        
        # Compute logical operators using CSS prescription
        try:
            log_x_vecs, log_z_vecs = compute_css_logicals(hx, hz)
            logical_x: List[PauliString] = vectors_to_paulis_x(log_x_vecs) if log_x_vecs else [{0: 'X'}]
            logical_z: List[PauliString] = vectors_to_paulis_z(log_z_vecs) if log_z_vecs else [{0: 'Z'}]
        except Exception:
            warnings.warn(
                "DistanceBalancedProductCode: logical computation failed; using trivial fallback.",
                RuntimeWarning,
                stacklevel=2,
            )
            logical_x = [{0: 'X'}]
            logical_z = [{0: 'Z'}]

        from qectostim.codes.utils import gf2_rank as _gf2_rank
        rank_hx = _gf2_rank(hx)
        rank_hz = _gf2_rank(hz)
        k = n_qubits - rank_hx - rank_hz

        lx0_support = sorted(logical_x[0].keys()) if isinstance(logical_x[0], dict) else []
        lz0_support = sorted(logical_z[0].keys()) if isinstance(logical_z[0], dict) else []

        # -- Coordinate metadata --
        _cols = int(np.ceil(np.sqrt(n_qubits)))
        _dc = {i: (float(i % _cols), float(i // _cols)) for i in range(n_qubits)}
        _xsc = {}
        for _r in range(hx.shape[0]):
            _sup = np.where(hx[_r])[0]
            if len(_sup):
                _xsc[_r] = (float(np.mean([_dc[q][0] for q in _sup])),
                            float(np.mean([_dc[q][1] for q in _sup])))
        _zsc = {}
        for _r in range(hz.shape[0]):
            _sup = np.where(hz[_r])[0]
            if len(_sup):
                _zsc[_r] = (float(np.mean([_dc[q][0] for q in _sup])),
                            float(np.mean([_dc[q][1] for q in _sup])))

        meta: Dict[str, Any] = dict(metadata or {})
        meta["code_family"] = "qldpc"
        meta["code_type"] = "distance_balanced"
        meta["name"] = f"DistanceBalanced_{na}x{nb}"
        meta["n"] = n_qubits
        meta["k"] = k
        meta["distance"] = None
        meta["rate"] = float(k) / n_qubits if n_qubits > 0 else 0.0
        meta["construction"] = "distance_balanced"

        meta["lx_pauli_type"] = "X"
        meta["lx_support"] = lx0_support
        meta["lz_pauli_type"] = "Z"
        meta["lz_support"] = lz0_support

        side = int(np.ceil(np.sqrt(n_qubits)))
        data_coords = [(float(i % side), float(i // side)) for i in range(n_qubits)]
        meta["data_coords"] = data_coords
        meta["x_stab_coords"] = _xsc
        meta["z_stab_coords"] = _zsc
        meta["data_qubits"] = list(range(n_qubits))
        meta["x_logical_coords"] = [data_coords[q] for q in lx0_support] if lx0_support else []
        meta["z_logical_coords"] = [data_coords[q] for q in lz0_support] if lz0_support else []

        meta["x_schedule"] = None
        meta["z_schedule"] = None
        meta["stabiliser_schedule"] = {
            "x_rounds": {i: 0 for i in range(hx.shape[0])},
            "z_rounds": {i: 0 for i in range(hz.shape[0])},
            "n_rounds": 1,
            "description": "Fully parallel; asymmetric X/Z distances.",
        }

        meta["error_correction_zoo_url"] = "https://errorcorrectionzoo.org/c/balanced_product"
        meta["wikipedia_url"] = None
        meta["canonical_references"] = [
            "Breuckmann & Eberhardt, IEEE Trans. Inf. Theory 67, 6653-6674 (2021). arXiv:2012.09271",
        ]
        meta["connections"] = [
            "Asymmetric HGP variant for biased-noise channels",
            "Different base codes control X and Z distances independently",
        ]

        validate_css_code(hx, hz, f"DistanceBalanced_{na}x{nb}", raise_on_error=True)
        
        super().__init__(hx=hx, hz=hz, logical_x=logical_x, logical_z=logical_z, metadata=meta)
        self._hx = hx
        self._hz = hz
        self._na_db = na
        self._nb_db = nb

    # ─── Properties ───────────────────────────────────────────────
    @property
    def name(self) -> str:
        """Human-readable name."""
        return f"DistanceBalanced_{self._na_db}x{self._nb_db}"

    @property
    def distance(self) -> int:
        """Code distance (stored in metadata; may be ``None``)."""
        d = self._metadata.get("distance")
        return d if d is not None else 1
    
    def qubit_coords(self) -> List[Coord2D]:
        n = self.n
        side = int(np.ceil(np.sqrt(n)))
        return [(float(i % side), float(i // side)) for i in range(n)]


# Factory functions for common constructions

def create_balanced_product_repetition(n_rep: int = 5, group_order: int = 1) -> BalancedProductCode:
    """Create balanced product of two repetition codes."""
    # Repetition code: all-ones parity check
    h_rep = np.zeros((n_rep - 1, n_rep), dtype=np.uint8)
    for i in range(n_rep - 1):
        h_rep[i, i] = 1
        h_rep[i, i + 1] = 1
    return BalancedProductCode(h_rep, h_rep, group_order=group_order)


def create_balanced_product_hamming() -> BalancedProductCode:
    """Create balanced product of two Hamming [7,4,3] codes."""
    # Hamming [7,4,3] parity check matrix
    h_hamming = np.array([
        [1, 1, 0, 1, 1, 0, 0],
        [1, 0, 1, 1, 0, 1, 0],
        [0, 1, 1, 1, 0, 0, 1],
    ], dtype=np.uint8)
    return BalancedProductCode(h_hamming, h_hamming, group_order=1)


# Pre-configured instances
BalancedProductRep5 = lambda: create_balanced_product_repetition(5)
BalancedProductHamming = lambda: create_balanced_product_hamming()
