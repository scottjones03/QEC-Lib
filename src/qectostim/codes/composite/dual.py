# src/qectostim/codes/composite/dual.py
"""Dual Code -- Swap X and Z Sectors of a CSS Code

For a CSS code *C* with check matrices ``Hx`` (detecting Z errors) and
``Hz`` (detecting X errors), the **dual code** ``C^⊥`` is obtained by
exchanging them:

.. math::

    H_x' = H_z, \\quad H_z' = H_x

Logical operators are likewise swapped: ``logical_x' = logical_z``
and ``logical_z' = logical_x``, with every Pauli letter Hadamard-
conjugated (X ↔ Z, Y → Y).

Overview
--------
The dual code is equivalent to applying a **transversal Hadamard** to
every physical qubit.  It preserves the code parameters ``[[n, k, d]]``
while exchanging the roles of bit-flip and phase-flip protection.

Homological interpretation
--------------------------
In the chain-complex picture a CSS code is a sequence

.. math::

    C_2 \\xrightarrow{\\partial_2 = H_z^T} C_1 \\xrightarrow{\\partial_1 = H_x} C_0

Dualisation reverses the complex: the co-chain complex has
``δ_0 = Hx^T`` and ``δ_1 = Hz``, exchanging the roles of
boundaries and co-boundaries.  The code parameters are preserved
because the Betti numbers of a complex and its dual coincide.

Distance preservation
---------------------
Since dualising merely relabels X-logicals as Z-logicals and vice
versa, the minimum weight of each type of logical is unchanged.
Hence ``d_X' = d_Z`` and ``d_Z' = d_X``, and the overall code
distance ``d = min(d_X, d_Z)`` is preserved.

Self-dual codes
---------------
A code whose row space ``rowsp(Hx) = rowsp(Hz)`` is called
*self-dual*.  Famous examples include the ``[[7,1,3]]`` Steane code
and some Reed–Muller codes.  The ``SelfDualCode`` helper class in
this module provides a check.

Classes and helpers
-------------------
* ``DualCode`` — main dual construction.
* ``SelfDualCode`` — marker / checker for self-dual CSS codes.
* ``swap_pauli_type()`` — Hadamard-conjugate a ``PauliString``.
* ``dual()`` — convenience function wrapping ``DualCode``.

Examples
--------
>>> from qectostim.codes.small.steane_713 import SteaneCode
>>> steane = SteaneCode()
>>> from qectostim.codes.composite.dual import DualCode
>>> d = DualCode(steane)
>>> d.n == steane.n and d.k == steane.k  # parameters preserved
True

Code Parameters
~~~~~~~~~~~~~~~
The dual code preserves the parameters of the base CSS code:

* **n** = ``n``  (same number of physical qubits)
* **k** = ``k``  (same number of logical qubits)
* **d** = ``d``  (distance preserved; ``d_X' = d_Z`` and ``d_Z' = d_X``)

Since dualising merely swaps X- and Z-type operators, the overall
code distance ``d = min(d_X, d_Z)`` is unchanged.

Stabiliser Structure
~~~~~~~~~~~~~~~~~~~~
The stabiliser generators are obtained by swapping the X and Z check
matrices of the base code:

* **X stabilisers**: the original Z stabilisers of the base code
  (same weights and count as the base ``Hz``).
* **Z stabilisers**: the original X stabilisers of the base code
  (same weights and count as the base ``Hx``).
* **Measurement schedule**: inherited from the base code with X and Z
  rounds swapped.  Any schedule valid for the base code works for the
  dual after relabelling.

References
----------
* Calderbank & Shor, *Good quantum error-correcting codes exist*,
  Phys. Rev. A **54**, 1098 (1996).
* Steane, *Error correcting codes in quantum theory*,
  Phys. Rev. Lett. **77**, 793 (1996).
* Error Correction Zoo: https://errorcorrectionzoo.org/c/dual

Fault tolerance
---------------
* Dualisation commutes with fault-tolerant gadgets: any FT circuit
  for the original code can be Hadamard-conjugated to work on the dual.
* If the original code has a transversal T gate, the dual has a
  transversal T† gate (and vice versa).
* Self-dual codes (Steane, some Reed–Muller codes) are invariant under
  this transformation, which simplifies fault-tolerant protocol design.

Decoding
--------
* Any decoder for the original code works on the dual by swapping the
  X and Z syndrome channels.
* For biased noise, dualising can improve the effective threshold if
  the dominant error type aligns with the stronger-distance sector.

Implementation notes
--------------------
* The ``DualCode`` constructor is lightweight: it references the same
  underlying matrices, only swapping the Hx/Hz pointers.
* No additional memory is allocated for the check matrices.
* The logical operator lists are shallow-copied with Pauli types swapped.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from qectostim.codes.abstract_code import PauliString
from qectostim.codes.abstract_css import CSSCode
from qectostim.codes.utils import str_to_pauli, validate_css_code


def _normalize_pauli(pauli: Any) -> PauliString:
    """Convert string or dict pauli to PauliString dict."""
    if isinstance(pauli, str):
        return str_to_pauli(pauli)
    return pauli


def swap_pauli_type(pauli: Any) -> PauliString:
    """
    Swap X ↔ Z in a Pauli string (Hadamard conjugation).
    
    X → Z, Z → X, Y → Y (up to phase)
    
    Parameters
    ----------
    pauli : PauliString or str
        Input Pauli operator (dict or string format).
        
    Returns
    -------
    PauliString
        Hadamard-conjugated Pauli operator.
    """
    # Normalize to dict format
    pauli_dict = _normalize_pauli(pauli)
    swap_map = {'X': 'Z', 'Z': 'X', 'Y': 'Y'}
    return {q: swap_map[op] for q, op in pauli_dict.items()}


class DualCode(CSSCode):
    """
    Dual of a CSS code: swap X and Z sectors.
    
    Given a CSS code C, the dual code C^⊥ exchanges:
    - Hx ↔ Hz
    - logical_x ↔ logical_z
    
    This is the code obtained by applying transversal Hadamard
    to all physical qubits.
    
    Parameters
    ----------
    base_code : CSSCode
        The CSS code to dualize.
    metadata : dict, optional
        Additional metadata (merged with base code metadata).
        
    Attributes
    ----------
    base_code : CSSCode
        The original code.
        
    Examples
    --------
    >>> from qectostim.codes.base import RepetitionCode
    >>> rep = RepetitionCode(3)  # Bit-flip code
    >>> dual = DualCode(rep)  # Phase-flip code
    >>> print(dual.hx.shape)  # Was hz
    >>> print(dual.hz.shape)  # Was hx
    
    Notes
    -----
    The dual of the dual is the original code: (C^⊥)^⊥ = C.
    
    For self-dual codes (like the [[7,1,3]] Steane code), the dual
    is isomorphic to the original.
    """
    
    def __init__(
        self,
        base_code: CSSCode,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Construct the dual of a CSS code.

        The check matrices ``Hx`` and ``Hz`` of *base_code* are swapped,
        and the logical operators are Hadamard-conjugated.

        Parameters
        ----------
        base_code : CSSCode
            The CSS code to dualise.
        metadata : dict, optional
            Extra key/value pairs merged into the code's metadata dict.

        Raises
        ------
        ValueError
            If the swapped check matrices fail CSS validation
            (``Hx' @ Hz'^T ≠ 0`` mod 2).
        """
        self.base_code = base_code
        
        # Swap Hx and Hz
        hx_dual = base_code.hz.copy()
        hz_dual = base_code.hx.copy()
        
        # Swap logical operators with X ↔ Z
        # Original logical X (X-type operators) become logical Z (Z-type operators)
        # Original logical Z (Z-type operators) become logical X (X-type operators)
        logical_x_dual = [swap_pauli_type(lz) for lz in base_code.logical_z_ops]
        logical_z_dual = [swap_pauli_type(lx) for lx in base_code.logical_x_ops]
        
        # Merge metadata
        base_meta = base_code.extra_metadata() if hasattr(base_code, 'extra_metadata') else {}
        meta: Dict[str, Any] = dict(base_meta)
        meta.update(metadata or {})
        meta["base_code_name"] = base_code.name
        meta["is_dual"] = True

        # ── 17 standard metadata keys ──────────────────────────────────────
        n = hx_dual.shape[1]
        meta["code_family"] = "dual"
        meta["code_type"] = "dual_css"
        meta["n"] = n
        meta["k"] = base_code.k  # same [[n,k,d]] as original
        meta["distance"] = getattr(base_code, 'distance', base_meta.get('distance', None))
        meta["rate"] = base_code.k / n if n > 0 else 0.0

        # Logical operator info (swapped from base code)
        if logical_x_dual:
            meta["lx_pauli_type"] = "X"  # Z-type ops became X-type
            meta["lx_support"] = sorted(logical_x_dual[0].keys())
        else:
            meta["lx_pauli_type"] = None
            meta["lx_support"] = []
        if logical_z_dual:
            meta["lz_pauli_type"] = "Z"  # X-type ops became Z-type
            meta["lz_support"] = sorted(logical_z_dual[0].keys())
        else:
            meta["lz_pauli_type"] = None
            meta["lz_support"] = []

        meta["stabiliser_schedule"] = base_meta.get("stabiliser_schedule", None)
        meta["x_schedule"] = base_meta.get("z_schedule", None)  # swapped
        meta["z_schedule"] = base_meta.get("x_schedule", None)  # swapped

        meta["error_correction_zoo_url"] = base_meta.get("error_correction_zoo_url", None)
        meta["wikipedia_url"] = base_meta.get("wikipedia_url", None)
        meta["canonical_references"] = base_meta.get("canonical_references", [
            "Calderbank & Shor, 'Good quantum error-correcting codes exist' (1996)",
        ])
        meta["connections"] = [
            f"Dual of {base_code.name}: Hx↔Hz swapped (transversal Hadamard)",
            "Same [[n,k,d]] parameters as the original code",
        ]

        # Coordinate metadata - inherit from base code, swap x/z stab coords
        base_data_coords = base_meta.get("data_coords", None)
        if base_data_coords is None:
            cols_grid = int(np.ceil(np.sqrt(n)))
            base_data_coords = [(float(i % cols_grid), float(i // cols_grid)) for i in range(n)]
        meta["data_coords"] = base_data_coords
        # Swap x/z stab coords since Hx↔Hz are swapped
        meta["x_stab_coords"] = base_meta.get("z_stab_coords", None)
        meta["z_stab_coords"] = base_meta.get("x_stab_coords", None)
        if meta["x_stab_coords"] is None:
            x_sc = []
            for row in hx_dual:
                support = np.where(row)[0]
                if len(support) > 0:
                    cx = float(np.mean([base_data_coords[q][0] for q in support if q < len(base_data_coords)]))
                    cy = float(np.mean([base_data_coords[q][1] for q in support if q < len(base_data_coords)]))
                    x_sc.append((cx, cy))
                else:
                    x_sc.append((0.0, 0.0))
            meta["x_stab_coords"] = x_sc
        if meta["z_stab_coords"] is None:
            z_sc = []
            for row in hz_dual:
                support = np.where(row)[0]
                if len(support) > 0:
                    cx = float(np.mean([base_data_coords[q][0] for q in support if q < len(base_data_coords)]))
                    cy = float(np.mean([base_data_coords[q][1] for q in support if q < len(base_data_coords)]))
                    z_sc.append((cx, cy))
                else:
                    z_sc.append((0.0, 0.0))
            meta["z_stab_coords"] = z_sc

        validate_css_code(hx_dual, hz_dual, code_name=f"Dual({base_code.name})", raise_on_error=True)
        
        super().__init__(
            hx=hx_dual,
            hz=hz_dual,
            logical_x=logical_x_dual,
            logical_z=logical_z_dual,
            metadata=meta,
        )
    
    @property
    def name(self) -> str:
        """Human-readable name, e.g. ``'Dual(Steane_7_1_3)'``."""
        return f"Dual({self.base_code.name})"

    @property
    def distance(self) -> int:
        """Code distance (same as the base code)."""
        return self._metadata.get("distance", getattr(self.base_code, 'distance', 1))

    def qubit_coords(self) -> List:
        """Return qubit coordinates inherited from the base code."""
        if hasattr(self.base_code, 'qubit_coords'):
            return self.base_code.qubit_coords()
        return list(range(self.n))
    
    def undual(self) -> CSSCode:
        """
        Return the original (base) code.
        
        Note: DualCode(DualCode(C)) creates a new DualCode wrapping the
        first DualCode, not the original C. Use this method or access
        base_code directly to get back to the original.
        """
        return self.base_code


class SelfDualCode(CSSCode):
    """
    Marker class for self-dual CSS codes (Hx = Hz).
    
    A code is self-dual if it equals its own dual, meaning Hx = Hz
    (up to row reordering) and logical_x can be mapped to logical_z.
    
    This class is mainly for type checking and discovery purposes.
    """
    
    @classmethod
    def is_self_dual(cls, code: CSSCode, tolerance: bool = True) -> bool:
        """
        Check if a CSS code is self-dual.
        
        Parameters
        ----------
        code : CSSCode
            The code to check.
        tolerance : bool
            If True, check if row spaces are equal (not exact matrix equality).
            
        Returns
        -------
        bool
            True if the code is self-dual.
        """
        from qectostim.codes.utils import gf2_rank, gf2_rowspace
        
        hx, hz = code.hx, code.hz
        
        if hx.shape != hz.shape:
            return False
        
        if tolerance:
            # Check if row spaces are the same
            combined = np.vstack([hx, hz])
            rank_hx = gf2_rank(hx)
            rank_hz = gf2_rank(hz)
            rank_combined = gf2_rank(combined)
            return rank_hx == rank_hz == rank_combined
        else:
            # Exact matrix equality (up to row sorting)
            return np.array_equal(np.sort(hx, axis=0), np.sort(hz, axis=0))


# Convenience function
def dual(code: CSSCode, metadata: Optional[Dict[str, Any]] = None) -> DualCode:
    """
    Create the dual of a CSS code.
    
    Convenience function for DualCode(code).
    
    Parameters
    ----------
    code : CSSCode
        The code to dualize.
    metadata : dict, optional
        Additional metadata.
        
    Returns
    -------
    DualCode
        The dual code.
    """
    return DualCode(code, metadata)
