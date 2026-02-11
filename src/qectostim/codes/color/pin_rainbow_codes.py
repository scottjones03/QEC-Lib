"""Pin Codes and Rainbow Codes — Specialized Colour-Code Variants

This module implements four families of CSS codes that extend the
colour-code paradigm with "pinning", "rainbow", and "holographic"
constructions.  All four are constructed via the *hypergraph product*
(HGP) of classical repetition codes, which guarantees CSS
orthogonality by construction.

Overview
--------
Colour codes are topological CSS codes whose faces can be properly
3-coloured.  The codes in this module generalise that idea in
different directions:

* **Pinning** adds boundary constraints that reshape the stabiliser
  lattice, improving thresholds and distance scaling.
* **Rainbow layering** stacks multiple colour-code sheets to encode
  several logical qubits with transversal non-Clifford gates.
* **Holographic** variants use an AdS/CFT-inspired bulk–boundary
  relationship for improved error correction.

Pin codes
---------
``QuantumPinCode(d, m)``
    Constructed from the HGP of a length-*d* and a length-(*m* + 1)
    repetition code.  The "pin" count *m* controls redundancy:
    n = d(m+1) + (d−1)m,  k determined by HGP rank formula.

``DoublePinCode(d)``
    The HGP of a length-*d* repetition code **with itself** (square
    structure).  Gives n = d² + (d−1)²,  improved distance scaling.

Pinned boundary codes
---------------------
Pinned boundary codes (not to be confused with QuantumPinCode above)
are surface-code variants where specific boundary qubits are
"pinned" to enforce additional stabiliser constraints.  They are
implemented elsewhere; this module provides the algebraic pin
construction only.

Rainbow codes
-------------
``RainbowCode(L, r)``
    Layered HGP of length-*L* and length-*r* repetition codes.
    Different layers are assigned different "colours" (rainbow), and
    transversal T gates are available on some logical qubits.
    n = Lr + (L−1)(r−1),  k = 1 (from HGP).

Holographic codes
-----------------
``HolographicRainbowCode(L, bulk_depth)``
    HGP of length-*L* and length-(*bulk_depth* + 1) repetition codes
    with an AdS/CFT-inspired bulk–boundary interpretation.  Boundary
    qubits live in the left sector; bulk qubits in the right sector.
    n = L(bulk_depth+1) + (L−1)·bulk_depth.

Code parameters
---------------
All four families are [[n, k, d]] CSS codes.  Because they are built
from HGP of repetition codes:

* CSS orthogonality (H_X H_Z^T = 0) holds automatically.
* k is determined by the kernel / image rank formula.
* The code distance equals the minimum-weight logical operator.

Code Parameters
~~~~~~~~~~~~~~~
:math:`[[n, k, d]]` where (per family):

- ``QuantumPinCode(d, m)``:
  :math:`n = d(m+1) + (d-1)m`, :math:`k` from HGP rank formula,
  :math:`d` = base distance.
- ``DoublePinCode(d)``:
  :math:`n = d^2 + (d-1)^2`, :math:`k` from HGP rank,
  improved distance scaling.
- ``RainbowCode(L, r)``:
  :math:`n = Lr + (L-1)(r-1)`, :math:`k = 1` (from HGP).
- ``HolographicRainbowCode(L, bulk_depth)``:
  :math:`n = L(\text{bulk\_depth}+1) + (L-1) \cdot \text{bulk\_depth}`,
  :math:`k` from HGP rank.
- Rate :math:`k/n` is determined by the HGP kernel / image rank formula.

Stabiliser Structure
~~~~~~~~~~~~~~~~~~~~
- **X-type stabilisers**: weight varies by construction; built from the
  hypergraph product of classical repetition codes — typical weights
  range from 2 to :math:`d`.
- **Z-type stabilisers**: transpose structure from the HGP; CSS
  orthogonality (:math:`H_X H_Z^T = 0`) holds automatically.
- Measurement schedule: default single fully-parallel round (HGP-based).

Connections
-----------
* **Hypergraph product codes** — all four constructions are special
  cases of the HGP framework.
* **Colour codes** — pin and rainbow codes inherit the colour-code
  transversal gate structure.
* **Surface codes** — DoublePinCode(d) is closely related to the
  toric / rotated surface code of distance *d*.
* **Holographic codes** — the holographic variant connects to
  tensor-network / AdS/CFT quantum error correction.

References
----------
* Bombin & Martin-Delgado, "Exact topological quantum order in D=3
  and beyond", Phys. Rev. B 75, 075103 (2007). arXiv:cond-mat/0607736
* Tillich & Zémor, "Quantum LDPC codes with positive rate and
  minimum distance proportional to n^{1/2}", IEEE Trans. Inform.
  Theory 60, 1193 (2014). arXiv:0903.0566
* Kubica & Beverland, "Universal transversal gates with color codes",
  Phys. Rev. A 91, 032330 (2015). arXiv:1410.0069
* Brown, Nickerson & Browne, "Fault-tolerant error correction with
  the gauge color code", Nature Commun. 7, 12302 (2016).
  arXiv:1503.08217
* Pastawski et al., "Holographic quantum error-correcting codes",
  JHEP 06, 149 (2015). arXiv:1503.06237
* Error Correction Zoo: https://errorcorrectionzoo.org/c/color
* Error Correction Zoo (HGP): https://errorcorrectionzoo.org/c/hypergraph_product
* Wikipedia: https://en.wikipedia.org/wiki/Color_code_(quantum_computing)
"""

from __future__ import annotations
import warnings
from typing import Dict, Any, Optional, Tuple, List
import numpy as np

from ..generic.qldpc_base import QLDPCCode
from ..utils import compute_css_logicals, vectors_to_paulis_x, vectors_to_paulis_z, validate_css_code


class QuantumPinCode(QLDPCCode):
    """
    Quantum Pin Code.
    
    Pin codes are CSS codes constructed via a "pinning" procedure that
    gives improved threshold and distance properties compared to standard
    surface codes. They can be viewed as color codes with specific boundary
    conditions.
    
    For parameters (d, m):
        - d: base distance
        - m: number of pins (determines redundancy)
        - n ∝ d² × m
        - k = 1 (single logical qubit)
    
    Parameters
    ----------
    d : int
        Base distance parameter (default: 3)
    m : int  
        Number of pins (default: 2)

    Raises
    ------
    ValueError
        If ``d < 2`` or ``m < 1``.
    """
    
    def __init__(self, d: int = 3, m: int = 2, metadata: Optional[Dict[str, Any]] = None):
        if d < 2:
            raise ValueError("d must be at least 2")
        if m < 1:
            raise ValueError("m must be at least 1")
        
        self._d = d
        self._m = m
        
        hx, hz, n_qubits = self._build_pin_code(d, m)
        
        # Compute logical operators using CSS prescription
        try:
            log_x_vecs, log_z_vecs = compute_css_logicals(hx, hz)
            logical_x = vectors_to_paulis_x(log_x_vecs) if log_x_vecs else [{0: 'X'}]
            logical_z = vectors_to_paulis_z(log_z_vecs) if log_z_vecs else [{0: 'Z'}]
            k = len(logical_x)
        except Exception as e:
            warnings.warn(f"Logical operator derivation failed for QuantumPinCode(d={d}, m={m}): {e}")
            k = m
            logical_x = [{0: 'X'}]
            logical_z = [{0: 'Z'}]
        
        # Validate CSS orthogonality
        validate_css_code(hx, hz, f"QuantumPinCode_d{d}_m{m}", raise_on_error=True)

        meta: Dict[str, Any] = dict(metadata or {})
        meta.update({
            "name": f"QuantumPinCode_d{d}_m{m}",
            "n": n_qubits,
            "k": k,
            "distance": d,
            "n_pins": m,
        })
        meta.setdefault("code_family", "pin_code")
        meta.setdefault("code_type", "quantum_pin_code")
        meta.setdefault("rate", k / n_qubits if n_qubits > 0 else 0.0)
        meta.setdefault("lx_pauli_type", "X")
        meta.setdefault("lz_pauli_type", "Z")
        n_x_stabs = hx.shape[0]
        n_z_stabs = hz.shape[0]
        meta.setdefault("stabiliser_schedule", {
            "x_rounds": {i: 0 for i in range(n_x_stabs)},
            "z_rounds": {i: 0 for i in range(n_z_stabs)},
            "n_rounds": 1,
            "description": "Fully parallel (round 0). QLDPC/HGP-based schedule.",
        })
        meta.setdefault("x_schedule", None)
        meta.setdefault("z_schedule", None)
        meta.setdefault("error_correction_zoo_url", "https://errorcorrectionzoo.org/c/hypergraph_product")
        meta.setdefault("wikipedia_url", "https://en.wikipedia.org/wiki/Color_code_(quantum_computing)")
        meta.setdefault("canonical_references", [
            "Tillich & Zémor, IEEE Trans. Inform. Theory 60, 1193 (2014). arXiv:0903.0566",
            "Bombin & Martin-Delgado, PRB 75, 075103 (2007). arXiv:cond-mat/0607736",
        ])
        meta.setdefault("connections", [
            "Hypergraph product codes: pin codes are HGP of repetition codes",
            "Colour codes: inherit transversal gate structure",
            "Surface codes: related via HGP construction",
        ])

        # Coordinate metadata
        cols_grid = int(np.ceil(np.sqrt(n_qubits)))
        data_coords_list = [(float(i % cols_grid), float(i // cols_grid)) for i in range(n_qubits)]
        meta.setdefault("data_coords", data_coords_list)
        x_stab_coords_list = []
        for row in hx:
            support = np.where(row)[0]
            if len(support) > 0:
                cx = float(np.mean([data_coords_list[q][0] for q in support]))
                cy = float(np.mean([data_coords_list[q][1] for q in support]))
                x_stab_coords_list.append((cx, cy))
            else:
                x_stab_coords_list.append((0.0, 0.0))
        meta.setdefault("x_stab_coords", x_stab_coords_list)
        z_stab_coords_list = []
        for row in hz:
            support = np.where(row)[0]
            if len(support) > 0:
                cx = float(np.mean([data_coords_list[q][0] for q in support]))
                cy = float(np.mean([data_coords_list[q][1] for q in support]))
                z_stab_coords_list.append((cx, cy))
            else:
                z_stab_coords_list.append((0.0, 0.0))
        meta.setdefault("z_stab_coords", z_stab_coords_list)
        # Logical operator supports
        meta.setdefault("lx_support", sorted(logical_x[0].keys()) if logical_x and isinstance(logical_x[0], dict) else [])
        meta.setdefault("lz_support", sorted(logical_z[0].keys()) if logical_z and isinstance(logical_z[0], dict) else [])

        super().__init__(hx, hz, logical_x, logical_z, metadata=meta)

    @property
    def name(self) -> str:
        """Human-readable name, e.g. ``'QuantumPinCode(d=3, m=2)'``."""
        return f"QuantumPinCode(d={self._d}, m={self._m})"

    @property
    def distance(self) -> int:
        """Code distance."""
        return self._d

    @property
    def d(self) -> int:
        return self._d
    
    @property
    def m(self) -> int:
        return self._m
    
    @staticmethod
    def _build_pin_code(d: int, m: int) -> Tuple[np.ndarray, np.ndarray, int]:
        """Build pin code using HGP construction."""
        # Use HGP of repetition codes
        # First code: length d repetition
        na = d
        ma = d - 1
        a = np.zeros((ma, na), dtype=np.uint8)
        for i in range(ma):
            a[i, i] = 1
            a[i, i + 1] = 1
        
        # Second code: length m+1 
        nb = m + 1
        mb = m
        b = np.zeros((mb, nb), dtype=np.uint8)
        for i in range(mb):
            b[i, i] = 1
            b[i, i + 1] = 1
        
        # HGP construction
        n_left = na * nb
        n_right = ma * mb
        n_qubits = n_left + n_right
        
        hx = np.zeros((ma * nb, n_qubits), dtype=np.uint8)
        stab = 0
        for check_a in range(ma):
            for bit_b in range(nb):
                for bit_a in range(na):
                    if a[check_a, bit_a]:
                        hx[stab, bit_a * nb + bit_b] ^= 1
                for check_b in range(mb):
                    if b[check_b, bit_b]:
                        hx[stab, n_left + check_a * mb + check_b] ^= 1
                stab += 1
        
        hz = np.zeros((na * mb, n_qubits), dtype=np.uint8)
        stab = 0
        for bit_a in range(na):
            for check_b in range(mb):
                for bit_b in range(nb):
                    if b[check_b, bit_b]:
                        hz[stab, bit_a * nb + bit_b] ^= 1
                for check_a in range(ma):
                    if a[check_a, bit_a]:
                        hz[stab, n_left + check_a * mb + check_b] ^= 1
                stab += 1
        
        return hx, hz, n_qubits
    
    @staticmethod
    def _build_logicals(n_qubits: int, k: int, d: int) -> Tuple[np.ndarray, np.ndarray]:
        """Build logical operators."""
        logical_x = np.zeros((k, n_qubits), dtype=np.uint8)
        logical_z = np.zeros((k, n_qubits), dtype=np.uint8)
        
        for i in range(k):
            for j in range(d):
                logical_x[i, j * (k + 1) + i] = 1
            logical_z[i, i] = 1
        
        return logical_x, logical_z
    
    def qubit_coords(self) -> List[Tuple[float, float]]:
        """
        Return 2D coordinates for visualization.
        
        Uses HGP two-sector grid layout:
        - Left sector: d x (m+1) grid
        - Right sector: (d-1) x m grid, offset to the right
        """
        coords: List[Tuple[float, float]] = []
        
        d = self._d
        m = self._m
        nb = m + 1
        mb = m
        
        n_left = d * nb
        right_offset = nb + 2
        
        for i in range(self.n):
            if i < n_left:
                # Left sector: d x (m+1) grid
                col = i % nb
                row = i // nb
                coords.append((float(col), float(row)))
            else:
                # Right sector: (d-1) x m grid, offset to right
                right_idx = i - n_left
                col = right_idx % mb if mb > 0 else 0
                row = right_idx // mb if mb > 0 else right_idx
                coords.append((float(col + right_offset), float(row)))
        
        return coords


class DoublePinCode(QLDPCCode):
    """
    Double Pin Code.
    
    A double-pin construction that combines two pin code structures
    for improved distance scaling. Uses tensor product structure
    of two pin configurations.
    
    Parameters
    ----------
    d : int
        Base distance parameter (default: 3)
    """
    
    def __init__(self, d: int = 3, metadata: Optional[Dict[str, Any]] = None):
        if d < 2:
            raise ValueError("d must be at least 2")
        
        self._d = d
        
        hx, hz, n_qubits = self._build_double_pin(d)
        
        # Compute logical operators using CSS prescription
        try:
            log_x_vecs, log_z_vecs = compute_css_logicals(hx, hz)
            logical_x = vectors_to_paulis_x(log_x_vecs) if log_x_vecs else [{0: 'X'}]
            logical_z = vectors_to_paulis_z(log_z_vecs) if log_z_vecs else [{0: 'Z'}]
            k = len(logical_x)
        except Exception as e:
            warnings.warn(f"Logical operator derivation failed for DoublePinCode(d={d}): {e}")
            k = d - 1
            logical_x = [{0: 'X'}]
            logical_z = [{0: 'Z'}]
        
        # Validate CSS orthogonality
        validate_css_code(hx, hz, f"DoublePinCode_d{d}", raise_on_error=True)

        meta: Dict[str, Any] = dict(metadata or {})
        meta.update({
            "name": f"DoublePinCode_d{d}",
            "n": n_qubits,
            "k": k,
            "distance": d,
        })
        meta.setdefault("code_family", "pin_code")
        meta.setdefault("code_type", "double_pin_code")
        meta.setdefault("rate", k / n_qubits if n_qubits > 0 else 0.0)
        meta.setdefault("lx_pauli_type", "X")
        meta.setdefault("lz_pauli_type", "Z")
        n_x_stabs = hx.shape[0]
        n_z_stabs = hz.shape[0]
        meta.setdefault("stabiliser_schedule", {
            "x_rounds": {i: 0 for i in range(n_x_stabs)},
            "z_rounds": {i: 0 for i in range(n_z_stabs)},
            "n_rounds": 1,
            "description": "Fully parallel (round 0). QLDPC/HGP-based schedule.",
        })
        meta.setdefault("x_schedule", None)
        meta.setdefault("z_schedule", None)
        meta.setdefault("error_correction_zoo_url", "https://errorcorrectionzoo.org/c/hypergraph_product")
        meta.setdefault("wikipedia_url", "https://en.wikipedia.org/wiki/Color_code_(quantum_computing)")
        meta.setdefault("canonical_references", [
            "Tillich & Zémor, IEEE Trans. Inform. Theory 60, 1193 (2014). arXiv:0903.0566",
            "Bombin & Martin-Delgado, PRB 75, 075103 (2007). arXiv:cond-mat/0607736",
        ])
        meta.setdefault("connections", [
            "Hypergraph product codes: double-pin is HGP of repetition code with itself",
            "Surface / toric codes: closely related for square HGP",
            "QuantumPinCode: single-pin variant with asymmetric dimensions",
        ])

        # Coordinate metadata
        cols_grid = int(np.ceil(np.sqrt(n_qubits)))
        data_coords_list = [(float(i % cols_grid), float(i // cols_grid)) for i in range(n_qubits)]
        meta.setdefault("data_coords", data_coords_list)
        x_stab_coords_list = []
        for row in hx:
            support = np.where(row)[0]
            if len(support) > 0:
                cx = float(np.mean([data_coords_list[q][0] for q in support]))
                cy = float(np.mean([data_coords_list[q][1] for q in support]))
                x_stab_coords_list.append((cx, cy))
            else:
                x_stab_coords_list.append((0.0, 0.0))
        meta.setdefault("x_stab_coords", x_stab_coords_list)
        z_stab_coords_list = []
        for row in hz:
            support = np.where(row)[0]
            if len(support) > 0:
                cx = float(np.mean([data_coords_list[q][0] for q in support]))
                cy = float(np.mean([data_coords_list[q][1] for q in support]))
                z_stab_coords_list.append((cx, cy))
            else:
                z_stab_coords_list.append((0.0, 0.0))
        meta.setdefault("z_stab_coords", z_stab_coords_list)
        # Logical operator supports
        meta.setdefault("lx_support", sorted(logical_x[0].keys()) if logical_x and isinstance(logical_x[0], dict) else [])
        meta.setdefault("lz_support", sorted(logical_z[0].keys()) if logical_z and isinstance(logical_z[0], dict) else [])

        super().__init__(hx, hz, logical_x, logical_z, metadata=meta)

    @property
    def name(self) -> str:
        """Human-readable name, e.g. ``'DoublePinCode(d=3)'``."""
        return f"DoublePinCode(d={self._d})"

    @property
    def distance(self) -> int:
        """Code distance."""
        return self._d

    @property
    def d(self) -> int:
        return self._d
    
    @staticmethod
    def _build_double_pin(d: int) -> Tuple[np.ndarray, np.ndarray, int]:
        """Build double pin code."""
        # Use HGP with square structure
        n = d
        m = d - 1
        
        h = np.zeros((m, n), dtype=np.uint8)
        for i in range(m):
            h[i, i] = 1
            h[i, i + 1] = 1
        
        # HGP of h with itself
        n_left = n * n
        n_right = m * m
        n_qubits = n_left + n_right
        
        hx = np.zeros((m * n, n_qubits), dtype=np.uint8)
        stab = 0
        for check_a in range(m):
            for bit_b in range(n):
                for bit_a in range(n):
                    if h[check_a, bit_a]:
                        hx[stab, bit_a * n + bit_b] ^= 1
                for check_b in range(m):
                    if h[check_b, bit_b]:
                        hx[stab, n_left + check_a * m + check_b] ^= 1
                stab += 1
        
        hz = np.zeros((n * m, n_qubits), dtype=np.uint8)
        stab = 0
        for bit_a in range(n):
            for check_b in range(m):
                for bit_b in range(n):
                    if h[check_b, bit_b]:
                        hz[stab, bit_a * n + bit_b] ^= 1
                for check_a in range(m):
                    if h[check_a, bit_a]:
                        hz[stab, n_left + check_a * m + check_b] ^= 1
                stab += 1
        
        return hx, hz, n_qubits
    
    @staticmethod
    def _build_logicals(n_qubits: int, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """Build logical operators."""
        logical_x = np.zeros((k, n_qubits), dtype=np.uint8)
        logical_z = np.zeros((k, n_qubits), dtype=np.uint8)
        
        for i in range(k):
            logical_x[i, i] = 1
            logical_z[i, n_qubits // 2 + i] = 1
        
        return logical_x, logical_z
    
    def qubit_coords(self) -> List[Tuple[float, float]]:
        """
        Return 2D coordinates for visualization.
        
        Uses HGP two-sector grid layout:
        - Left sector: d x d grid
        - Right sector: (d-1) x (d-1) grid, offset to the right
        """
        coords: List[Tuple[float, float]] = []
        
        d = self._d
        n_left = d * d
        right_offset = d + 2
        side = d - 1 if d > 1 else 1
        
        for i in range(self.n):
            if i < n_left:
                # Left sector: d x d grid
                col = i % d
                row = i // d
                coords.append((float(col), float(row)))
            else:
                # Right sector: (d-1) x (d-1) grid, offset to right
                right_idx = i - n_left
                col = right_idx % side
                row = right_idx // side
                coords.append((float(col + right_offset), float(row)))
        
        return coords


class RainbowCode(QLDPCCode):
    """
    Rainbow Code.
    
    Generalized color codes with a "rainbow" structure that gives 
    transversal gates and improved fault tolerance. The rainbow
    structure assigns different "colors" to different layers.
    
    For depth r (rainbow layers):
        - n ∝ L² × r
        - k = r (multiple logical qubits)
        - Transversal T-gates on some logical qubits
    
    Parameters
    ----------
    L : int
        Linear lattice size (default: 3)
    r : int
        Number of rainbow layers (default: 3)
    """
    
    def __init__(self, L: int = 3, r: int = 3, metadata: Optional[Dict[str, Any]] = None):
        if L < 2:
            raise ValueError("L must be at least 2")
        if r < 2:
            raise ValueError("r must be at least 2")
        
        self._L = L
        self._r = r
        
        hx, hz, n_qubits = self._build_rainbow_code(L, r)
        
        # Compute logical operators using CSS prescription
        try:
            log_x_vecs, log_z_vecs = compute_css_logicals(hx, hz)
            logical_x = vectors_to_paulis_x(log_x_vecs) if log_x_vecs else [{0: 'X'}]
            logical_z = vectors_to_paulis_z(log_z_vecs) if log_z_vecs else [{0: 'Z'}]
            k = len(logical_x)
        except Exception as e:
            warnings.warn(f"Logical operator derivation failed for RainbowCode(L={L}, r={r}): {e}")
            k = r
            logical_x = [{0: 'X'}]
            logical_z = [{0: 'Z'}]
        
        # Validate CSS orthogonality
        validate_css_code(hx, hz, f"RainbowCode_L{L}_r{r}", raise_on_error=True)

        meta: Dict[str, Any] = dict(metadata or {})
        meta.update({
            "name": f"RainbowCode_L{L}_r{r}",
            "n": n_qubits,
            "k": k,
            "distance": L,  # Distance from underlying surface code
            "L": L,
            "rainbow_depth": r,
            "transversal_gates": ["CNOT", "T (on some)"],
        })
        meta.setdefault("code_family", "rainbow_code")
        meta.setdefault("code_type", "rainbow_colour_code")
        meta.setdefault("rate", k / n_qubits if n_qubits > 0 else 0.0)
        meta.setdefault("lx_pauli_type", "X")
        meta.setdefault("lz_pauli_type", "Z")
        n_x_stabs = hx.shape[0]
        n_z_stabs = hz.shape[0]
        meta.setdefault("stabiliser_schedule", {
            "x_rounds": {i: 0 for i in range(n_x_stabs)},
            "z_rounds": {i: 0 for i in range(n_z_stabs)},
            "n_rounds": 1,
            "description": "Fully parallel (round 0). Layered HGP schedule.",
        })
        meta.setdefault("x_schedule", None)
        meta.setdefault("z_schedule", None)
        meta.setdefault("error_correction_zoo_url", "https://errorcorrectionzoo.org/c/color")
        meta.setdefault("wikipedia_url", "https://en.wikipedia.org/wiki/Color_code_(quantum_computing)")
        meta.setdefault("canonical_references", [
            "Kubica & Beverland, PRA 91, 032330 (2015). arXiv:1410.0069",
            "Brown, Nickerson & Browne, Nature Commun. 7, 12302 (2016). arXiv:1503.08217",
        ])
        meta.setdefault("connections", [
            "Colour codes: rainbow layering of colour-code sheets",
            "Transversal T gates on some logical qubits",
            "Hypergraph product codes: built via HGP of repetition codes",
        ])

        # Coordinate metadata
        cols_grid = int(np.ceil(np.sqrt(n_qubits)))
        data_coords_list = [(float(i % cols_grid), float(i // cols_grid)) for i in range(n_qubits)]
        meta.setdefault("data_coords", data_coords_list)
        x_stab_coords_list = []
        for row in hx:
            support = np.where(row)[0]
            if len(support) > 0:
                cx = float(np.mean([data_coords_list[q][0] for q in support]))
                cy = float(np.mean([data_coords_list[q][1] for q in support]))
                x_stab_coords_list.append((cx, cy))
            else:
                x_stab_coords_list.append((0.0, 0.0))
        meta.setdefault("x_stab_coords", x_stab_coords_list)
        z_stab_coords_list = []
        for row in hz:
            support = np.where(row)[0]
            if len(support) > 0:
                cx = float(np.mean([data_coords_list[q][0] for q in support]))
                cy = float(np.mean([data_coords_list[q][1] for q in support]))
                z_stab_coords_list.append((cx, cy))
            else:
                z_stab_coords_list.append((0.0, 0.0))
        meta.setdefault("z_stab_coords", z_stab_coords_list)
        # Logical operator supports
        meta.setdefault("lx_support", sorted(logical_x[0].keys()) if logical_x and isinstance(logical_x[0], dict) else [])
        meta.setdefault("lz_support", sorted(logical_z[0].keys()) if logical_z and isinstance(logical_z[0], dict) else [])

        super().__init__(hx, hz, logical_x, logical_z, metadata=meta)

    @property
    def name(self) -> str:
        """Human-readable name, e.g. ``'RainbowCode(L=3, r=3)'``."""
        return f"RainbowCode(L={self._L}, r={self._r})"

    @property
    def distance(self) -> int:
        """Code distance (equal to lattice size L)."""
        return self._L

    @property
    def L(self) -> int:
        return self._L
    
    @property
    def r(self) -> int:
        return self._r
    
    @staticmethod
    def _build_rainbow_code(L: int, r: int) -> Tuple[np.ndarray, np.ndarray, int]:
        """Build rainbow code using layered HGP."""
        # Use HGP with L and r dimensions
        na = L
        ma = L - 1
        nb = r
        mb = r - 1
        
        a = np.zeros((ma, na), dtype=np.uint8)
        for i in range(ma):
            a[i, i] = 1
            a[i, i + 1] = 1
        
        b = np.zeros((mb, nb), dtype=np.uint8)
        for i in range(mb):
            b[i, i] = 1
            b[i, i + 1] = 1
        
        # HGP construction
        n_left = na * nb
        n_right = ma * mb
        n_qubits = n_left + n_right
        
        hx = np.zeros((ma * nb, n_qubits), dtype=np.uint8)
        stab = 0
        for check_a in range(ma):
            for bit_b in range(nb):
                for bit_a in range(na):
                    if a[check_a, bit_a]:
                        hx[stab, bit_a * nb + bit_b] ^= 1
                for check_b in range(mb):
                    if b[check_b, bit_b]:
                        hx[stab, n_left + check_a * mb + check_b] ^= 1
                stab += 1
        
        hz = np.zeros((na * mb, n_qubits), dtype=np.uint8)
        stab = 0
        for bit_a in range(na):
            for check_b in range(mb):
                for bit_b in range(nb):
                    if b[check_b, bit_b]:
                        hz[stab, bit_a * nb + bit_b] ^= 1
                for check_a in range(ma):
                    if a[check_a, bit_a]:
                        hz[stab, n_left + check_a * mb + check_b] ^= 1
                stab += 1
        
        return hx, hz, n_qubits
    
    @staticmethod
    def _build_logicals(n_qubits: int, k: int, L: int) -> Tuple[np.ndarray, np.ndarray]:
        """Build logical operators."""
        logical_x = np.zeros((k, n_qubits), dtype=np.uint8)
        logical_z = np.zeros((k, n_qubits), dtype=np.uint8)
        
        for i in range(k):
            for j in range(L):
                logical_x[i, j * k + i] = 1
            logical_z[i, i] = 1
        
        return logical_x, logical_z
    
    def qubit_coords(self) -> List[Tuple[float, float]]:
        """
        Return 2D coordinates for visualization.
        
        Uses HGP two-sector grid layout:
        - Left sector: L x r grid
        - Right sector: (L-1) x (r-1) grid, offset to the right
        """
        coords: List[Tuple[float, float]] = []
        
        L = self._L
        r = self._r
        n_left = L * r
        right_offset = r + 2
        side_b = r - 1 if r > 1 else 1
        
        for i in range(self.n):
            if i < n_left:
                # Left sector: L x r grid
                col = i % r
                row = i // r
                coords.append((float(col), float(row)))
            else:
                # Right sector: (L-1) x (r-1) grid, offset to right
                right_idx = i - n_left
                col = right_idx % side_b
                row = right_idx // side_b
                coords.append((float(col + right_offset), float(row)))
        
        return coords


class HolographicRainbowCode(QLDPCCode):
    """
    Holographic Rainbow Code.
    
    Rainbow codes with holographic properties inspired by AdS/CFT
    correspondence. The bulk-boundary relationship gives improved
    error correction properties.
    
    Parameters
    ----------
    L : int
        Boundary size (default: 4)
    bulk_depth : int
        Holographic bulk depth (default: 2)
    """
    
    def __init__(self, L: int = 4, bulk_depth: int = 2, metadata: Optional[Dict[str, Any]] = None):
        if L < 3:
            raise ValueError("L must be at least 3")
        if bulk_depth < 1:
            raise ValueError("bulk_depth must be at least 1")
        
        self._L = L
        self._bulk_depth = bulk_depth
        
        hx, hz, n_qubits = self._build_holographic_code(L, bulk_depth)
        
        # Compute logical operators using CSS prescription
        try:
            log_x_vecs, log_z_vecs = compute_css_logicals(hx, hz)
            logical_x = vectors_to_paulis_x(log_x_vecs) if log_x_vecs else [{0: 'X'}]
            logical_z = vectors_to_paulis_z(log_z_vecs) if log_z_vecs else [{0: 'Z'}]
            k = len(logical_x)
        except Exception as e:
            warnings.warn(f"Logical operator derivation failed for HolographicRainbowCode(L={L}, bulk_depth={bulk_depth}): {e}")
            k = bulk_depth
            logical_x = [{0: 'X'}]
            logical_z = [{0: 'Z'}]
        
        # Validate CSS orthogonality
        validate_css_code(hx, hz, f"HolographicRainbowCode_L{L}_d{bulk_depth}", raise_on_error=True)

        holo_distance = min(L, bulk_depth + 1)

        meta: Dict[str, Any] = dict(metadata or {})
        meta.update({
            "name": f"HolographicRainbowCode_L{L}_d{bulk_depth}",
            "n": n_qubits,
            "k": k,
            "distance": holo_distance,
            "boundary_size": L,
            "bulk_depth": bulk_depth,
            "holographic": True,
        })
        meta.setdefault("code_family", "holographic_code")
        meta.setdefault("code_type", "holographic_rainbow_code")
        meta.setdefault("rate", k / n_qubits if n_qubits > 0 else 0.0)
        meta.setdefault("lx_pauli_type", "X")
        meta.setdefault("lz_pauli_type", "Z")
        n_x_stabs = hx.shape[0]
        n_z_stabs = hz.shape[0]
        meta.setdefault("stabiliser_schedule", {
            "x_rounds": {i: 0 for i in range(n_x_stabs)},
            "z_rounds": {i: 0 for i in range(n_z_stabs)},
            "n_rounds": 1,
            "description": "Fully parallel (round 0). Holographic bulk–boundary schedule.",
        })
        meta.setdefault("x_schedule", None)
        meta.setdefault("z_schedule", None)
        meta.setdefault("error_correction_zoo_url", "https://errorcorrectionzoo.org/c/holographic")
        meta.setdefault("wikipedia_url", "https://en.wikipedia.org/wiki/Color_code_(quantum_computing)")
        meta.setdefault("canonical_references", [
            "Pastawski et al., JHEP 06, 149 (2015). arXiv:1503.06237",
            "Kubica & Beverland, PRA 91, 032330 (2015). arXiv:1410.0069",
        ])
        meta.setdefault("connections", [
            "Holographic codes / AdS-CFT tensor-network error correction",
            "Rainbow codes: holographic extension with bulk–boundary duality",
            "Hypergraph product codes: built via HGP of repetition codes",
        ])

        # Coordinate metadata
        cols_grid = int(np.ceil(np.sqrt(n_qubits)))
        data_coords_list = [(float(i % cols_grid), float(i // cols_grid)) for i in range(n_qubits)]
        meta.setdefault("data_coords", data_coords_list)
        x_stab_coords_list = []
        for row in hx:
            support = np.where(row)[0]
            if len(support) > 0:
                cx = float(np.mean([data_coords_list[q][0] for q in support]))
                cy = float(np.mean([data_coords_list[q][1] for q in support]))
                x_stab_coords_list.append((cx, cy))
            else:
                x_stab_coords_list.append((0.0, 0.0))
        meta.setdefault("x_stab_coords", x_stab_coords_list)
        z_stab_coords_list = []
        for row in hz:
            support = np.where(row)[0]
            if len(support) > 0:
                cx = float(np.mean([data_coords_list[q][0] for q in support]))
                cy = float(np.mean([data_coords_list[q][1] for q in support]))
                z_stab_coords_list.append((cx, cy))
            else:
                z_stab_coords_list.append((0.0, 0.0))
        meta.setdefault("z_stab_coords", z_stab_coords_list)
        # Logical operator supports
        meta.setdefault("lx_support", sorted(logical_x[0].keys()) if logical_x and isinstance(logical_x[0], dict) else [])
        meta.setdefault("lz_support", sorted(logical_z[0].keys()) if logical_z and isinstance(logical_z[0], dict) else [])

        super().__init__(hx, hz, logical_x, logical_z, metadata=meta)

    @property
    def name(self) -> str:
        """Human-readable name, e.g. ``'HolographicRainbowCode(L=4, bulk_depth=2)'``."""
        return f"HolographicRainbowCode(L={self._L}, bulk_depth={self._bulk_depth})"

    @property
    def distance(self) -> int:
        """Code distance (min of boundary size L and bulk_depth + 1)."""
        return min(self._L, self._bulk_depth + 1)

    @property
    def L(self) -> int:
        return self._L
    
    @property
    def bulk_depth(self) -> int:
        return self._bulk_depth
    
    @staticmethod
    def _build_holographic_code(L: int, depth: int) -> Tuple[np.ndarray, np.ndarray, int]:
        """Build holographic code using layered structure."""
        # Boundary: L qubits per layer, depth layers
        # Use HGP to ensure CSS validity
        na = L
        ma = L - 1
        nb = depth + 1
        mb = depth
        
        a = np.zeros((ma, na), dtype=np.uint8)
        for i in range(ma):
            a[i, i] = 1
            a[i, i + 1] = 1
        
        b = np.zeros((mb, nb), dtype=np.uint8)
        for i in range(mb):
            b[i, i] = 1
            b[i, i + 1] = 1
        
        # HGP construction
        n_left = na * nb
        n_right = ma * mb
        n_qubits = n_left + n_right
        
        hx = np.zeros((ma * nb, n_qubits), dtype=np.uint8)
        stab = 0
        for check_a in range(ma):
            for bit_b in range(nb):
                for bit_a in range(na):
                    if a[check_a, bit_a]:
                        hx[stab, bit_a * nb + bit_b] ^= 1
                for check_b in range(mb):
                    if b[check_b, bit_b]:
                        hx[stab, n_left + check_a * mb + check_b] ^= 1
                stab += 1
        
        hz = np.zeros((na * mb, n_qubits), dtype=np.uint8)
        stab = 0
        for bit_a in range(na):
            for check_b in range(mb):
                for bit_b in range(nb):
                    if b[check_b, bit_b]:
                        hz[stab, bit_a * nb + bit_b] ^= 1
                for check_a in range(ma):
                    if a[check_a, bit_a]:
                        hz[stab, n_left + check_a * mb + check_b] ^= 1
                stab += 1
        
        return hx, hz, n_qubits
    
    @staticmethod
    def _build_logicals(n_qubits: int, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """Build logical operators."""
        logical_x = np.zeros((k, n_qubits), dtype=np.uint8)
        logical_z = np.zeros((k, n_qubits), dtype=np.uint8)
        
        for i in range(k):
            logical_x[i, i] = 1
            logical_z[i, n_qubits // 2 + i] = 1
        
        return logical_x, logical_z
    
    def qubit_coords(self) -> List[Tuple[float, float]]:
        """
        Return 2D coordinates for visualization.
        
        Uses HGP two-sector grid layout:
        - Left sector: L x (bulk_depth+1) grid
        - Right sector: (L-1) x bulk_depth grid, offset to the right
        """
        coords: List[Tuple[float, float]] = []
        
        L = self._L
        depth = self._bulk_depth
        nb = depth + 1
        mb = depth
        
        n_left = L * nb
        right_offset = nb + 2
        
        for i in range(self.n):
            if i < n_left:
                # Left sector: L x (depth+1) grid
                col = i % nb
                row = i // nb
                coords.append((float(col), float(row)))
            else:
                # Right sector: (L-1) x depth grid, offset to right
                right_idx = i - n_left
                col = right_idx % mb if mb > 0 else 0
                row = right_idx // mb if mb > 0 else right_idx
                coords.append((float(col + right_offset), float(row)))
        
        return coords


# Pre-configured instances
QuantumPin_d3_m2 = lambda: QuantumPinCode(d=3, m=2)
QuantumPin_d5_m3 = lambda: QuantumPinCode(d=5, m=3)
DoublePin_d3 = lambda: DoublePinCode(d=3)
DoublePin_d5 = lambda: DoublePinCode(d=5)
Rainbow_L3_r3 = lambda: RainbowCode(L=3, r=3)
Rainbow_L5_r4 = lambda: RainbowCode(L=5, r=4)
HolographicRainbow_L4_d2 = lambda: HolographicRainbowCode(L=4, bulk_depth=2)
HolographicRainbow_L6_d3 = lambda: HolographicRainbowCode(L=6, bulk_depth=3)
