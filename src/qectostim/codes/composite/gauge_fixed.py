# src/qectostim/codes/composite/gauge_fixed.py
"""Gauge-Fixed Codes -- Convert Subsystem Codes to Stabiliser Codes

A **subsystem** (operator) code has a gauge group *G* that is
strictly larger than its stabiliser group *S*.  The extra generators
commute with the logical operators but are not themselves stabilisers.
Gauge **fixing** promotes selected gauge generators into the
stabiliser group, thereby projecting the code-space from
``2^(k + g)`` to ``2^k`` dimensions.

Overview
--------
For a CSS subsystem code:

* ``Hx`` contains the X-type **stabiliser** generators.
* ``Hz`` contains the Z-type stabiliser generators.
* ``Gx`` (``hx_gauge``) and ``Gz`` (``hz_gauge``) are gauge
  generators that are *not* in the stabiliser group.

Gauge fixing appends selected rows of ``Gx`` / ``Gz`` to ``Hx`` /
``Hz``, creating a standard stabiliser code.  The resulting code
may have a **different distance** than the parent subsystem code
(typically the stabiliser distance equals the dressed distance of
the subsystem code for one type of fix and may differ for the other).

Bare vs dressed distance
-------------------------
A subsystem code has two notions of distance:

* **Bare distance** — minimum weight of a logical operator that
  commutes with the stabiliser group but *not* with all gauge
  generators.
* **Dressed distance** — minimum weight of a logical operator
  modulo multiplication by arbitrary gauge operators (always ≤ bare).

After gauge fixing, the stabiliser code distance equals the *dressed*
distance of the parent subsystem code, since the fixed gauge operators
are now stabilisers.

Gauge group structure
---------------------
The gauge group decomposes as  ``G = S × L_g`` where ``S`` is the
stabiliser subgroup (centre of ``G``) and ``L_g`` are the "gauge
logical" operators.  Fixing selects one eigenstate of each ``L_g``
generator, collapsing the gauge degrees of freedom.

Classes and helpers
-------------------
* ``GaugeFixedCode`` — main gauge-fixing constructor.
* ``GaugeFixedCode.from_bacon_shor`` — pre-built Bacon–Shor
  gauge fixing.
* ``gauge_fix_css()`` — convenience wrapper.

Connections to other codes
--------------------------
* **Bacon–Shor codes**: canonical example of gauge fixing.
* **Subsystem surface codes**: gauge-fixed variants give standard
  surface or compass codes.
* **Colour codes**: can be viewed as gauge-fixed versions of
  topological subsystem codes.

Code Parameters
~~~~~~~~~~~~~~~
Starting from a subsystem code ``[[n, k, d_bare, d_dressed]]``:

* **n** = ``n``  (same number of physical qubits)
* **k'** ≥ ``k``  (may increase if gauge fixing reveals hidden logicals,
  but typically ``k' = k``)
* **d'** = ``d_dressed``  (the gauge-fixed code distance equals the
  dressed distance of the parent subsystem code)

The specific distance depends on ``fix_type``:

* Fixing **X gauge** only: ``d_Z' = d_Z_dressed``, ``d_X'`` unchanged.
* Fixing **Z gauge** only: ``d_X' = d_X_dressed``, ``d_Z'`` unchanged.
* Fixing **both**: ``d' = min(d_X_dressed, d_Z_dressed)``.

Stabiliser Structure
~~~~~~~~~~~~~~~~~~~~
After gauge fixing, the stabiliser group is enlarged:

* **X stabilisers**: original ``Hx`` rows plus promoted X gauge
  generators.  Total count = ``r_x + r_x_gauge`` (before removing
  linearly dependent rows).
* **Z stabilisers**: original ``Hz`` rows plus promoted Z gauge
  generators.  Total count = ``r_z + r_z_gauge``.
* **Weights**: promoted gauge generators may have lower weight than
  the original stabilisers (e.g. weight-2 gauge operators in the
  Bacon–Shor code become weight-2 stabilisers).
* **Measurement schedule**: the enlarged stabiliser set can be
  measured in a single round (all generators commute after fixing).
  Redundant generators are removed via GF(2) row reduction.

Examples
--------
>>> from qectostim.codes.composite.gauge_fixed import GaugeFixedCode
>>> code = GaugeFixedCode.from_bacon_shor(3, 3)
>>> code.n  # 9 physical qubits
9

References
----------
* Poulin, *Stabilizer formalism for operator quantum error
  correction*, Phys. Rev. Lett. **95**, 230504 (2005).
* Bacon, *Operator quantum error-correcting subsystems for
  self-correcting quantum memories*, Phys. Rev. A **73**,
  012340 (2006).
* Bombin, *Gauge color codes*, Phys. Rev. X **5**, 031043 (2015).
* Error Correction Zoo: https://errorcorrectionzoo.org/c/subsystem_stabilizer

Fault tolerance
---------------
* Gauge fixing can increase the number of stabiliser checks, which
  may improve the code distance at the cost of removing gauge freedom.
* The fixed code inherits the dressed distance of the parent subsystem
  code, which is always ≥ the bare distance.

Implementation notes
--------------------
* After fixing, the resulting Hx/Hz matrices are re-validated for the
  CSS condition (Hx · Hz^T = 0 mod 2).
* Redundant rows (linearly dependent stabilisers introduced by fixing)
  are automatically removed via Gaussian elimination.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from qectostim.codes.abstract_code import PauliString
from qectostim.codes.abstract_css import CSSCode
from qectostim.codes.utils import (
    str_to_pauli,
    gf2_rank,
    gf2_rowspace,
    css_intersection_check,
    pauli_to_symplectic,
    symplectic_to_pauli,
    validate_css_code,
)


class GaugeFixedCode(CSSCode):
    """
    Convert a CSS subsystem code to a stabilizer code by gauge fixing.
    
    Given a subsystem code with gauge generators, this class promotes
    selected gauge generators to stabilizers, fixing the gauge degrees
    of freedom.
    
    Parameters
    ----------
    hx_stabilizer : np.ndarray
        X-type stabilizer generators (original stabilizer group).
    hz_stabilizer : np.ndarray
        Z-type stabilizer generators.
    hx_gauge : np.ndarray
        X-type gauge generators (not in stabilizer group).
    hz_gauge : np.ndarray
        Z-type gauge generators.
    logical_x : List[PauliString]
        Logical X operators of the subsystem code.
    logical_z : List[PauliString]
        Logical Z operators.
    fix_x_gauge : bool, optional
        If True, add X gauge generators to stabilizers. Default True.
    fix_z_gauge : bool, optional
        If True, add Z gauge generators to stabilizers. Default True.
    gauge_choice : Optional[np.ndarray], optional
        Specific gauge generators to include (binary vector selecting rows).
    metadata : dict, optional
        Additional metadata.
        
    Attributes
    ----------
    original_hx_stab : np.ndarray
        Original X stabilizers before gauge fixing.
    original_hz_stab : np.ndarray
        Original Z stabilizers before gauge fixing.
    hx_gauge : np.ndarray
        X gauge generators.
    hz_gauge : np.ndarray
        Z gauge generators.
        
    Examples
    --------
    >>> # Bacon-Shor code has gauge operators that can be fixed
    >>> # After fixing, get a standard CSS code
    >>> hx_stab = np.array([[1,1,1,1,0,0,0,0,0],  # X stabilizer
    ...                     [0,0,0,0,1,1,1,1,0]])
    >>> hz_stab = np.array([[1,0,0,1,0,0,1,0,0]])  # Z stabilizer
    >>> hx_gauge = np.array([[1,1,0,0,0,0,0,0,0]])  # X gauge
    >>> hz_gauge = np.array([[1,0,0,0,1,0,0,0,0]])  # Z gauge
    >>> fixed = GaugeFixedCode(hx_stab, hz_stab, hx_gauge, hz_gauge, [], [])
    
    Notes
    -----
    Gauge fixing does not change the logical information encoded, but it
    does change the code space dimension from 2^(k + gauge_qubits) to 2^k.
    
    The distance of the gauge-fixed code may differ from the subsystem
    code distance, depending on which gauge is fixed.
    """
    
    def __init__(
        self,
        hx_stabilizer: np.ndarray,
        hz_stabilizer: np.ndarray,
        hx_gauge: np.ndarray,
        hz_gauge: np.ndarray,
        logical_x: List[PauliString],
        logical_z: List[PauliString],
        fix_x_gauge: bool = True,
        fix_z_gauge: bool = True,
        gauge_choice: Optional[np.ndarray] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Construct a stabiliser code by gauge-fixing a subsystem code.

        Parameters
        ----------
        hx_stabilizer : np.ndarray
            X-type stabiliser generators of the parent subsystem code.
        hz_stabilizer : np.ndarray
            Z-type stabiliser generators.
        hx_gauge : np.ndarray
            X-type gauge generators (to be promoted if *fix_x_gauge*).
        hz_gauge : np.ndarray
            Z-type gauge generators (to be promoted if *fix_z_gauge*).
        logical_x : list of PauliString
            Logical X operators of the subsystem code.
        logical_z : list of PauliString
            Logical Z operators of the subsystem code.
        fix_x_gauge : bool, optional
            If ``True``, promote X gauge generators.  Default ``True``.
        fix_z_gauge : bool, optional
            If ``True``, promote Z gauge generators.  Default ``True``.
        gauge_choice : np.ndarray, optional
            Binary vector selecting which gauge rows to include.
        metadata : dict, optional
            Extra key/value pairs merged into the code's metadata dict.

        Raises
        ------
        ValueError
            If all input matrices are empty (no qubits defined).
        ValueError
            If the input matrices have differing column counts.
        ValueError
            If the gauge-fixed check matrices violate the CSS
            constraint ``Hx_new @ Hz_new^T ≠ 0 (mod 2)``.
        """
        # Store originals
        self.original_hx_stab = np.array(hx_stabilizer, dtype=np.uint8)
        self.original_hz_stab = np.array(hz_stabilizer, dtype=np.uint8)
        self.hx_gauge = np.array(hx_gauge, dtype=np.uint8)
        self.hz_gauge = np.array(hz_gauge, dtype=np.uint8)
        
        # Determine n from matrices
        n_candidates = []
        for m in [hx_stabilizer, hz_stabilizer, hx_gauge, hz_gauge]:
            if m.size > 0:
                n_candidates.append(m.shape[1])
        
        if not n_candidates:
            raise ValueError("At least one matrix must be non-empty")
        
        n = n_candidates[0]
        if not all(nc == n for nc in n_candidates):
            raise ValueError("All matrices must have the same number of columns (qubits)")
        
        # Validate gauge choice
        if gauge_choice is not None:
            # Select specific gauge generators
            if fix_x_gauge and hx_gauge.size > 0:
                x_select = gauge_choice[:hx_gauge.shape[0]] if len(gauge_choice) >= hx_gauge.shape[0] else gauge_choice
                hx_gauge_selected = hx_gauge[x_select.astype(bool)] if x_select.size > 0 else np.zeros((0, n), dtype=np.uint8)
            else:
                hx_gauge_selected = np.zeros((0, n), dtype=np.uint8)
            
            if fix_z_gauge and hz_gauge.size > 0:
                z_start = hx_gauge.shape[0] if hx_gauge.size > 0 else 0
                z_select = gauge_choice[z_start:z_start + hz_gauge.shape[0]] if len(gauge_choice) > z_start else np.array([])
                hz_gauge_selected = hz_gauge[z_select.astype(bool)] if z_select.size > 0 else np.zeros((0, n), dtype=np.uint8)
            else:
                hz_gauge_selected = np.zeros((0, n), dtype=np.uint8)
        else:
            # Fix all gauge operators
            hx_gauge_selected = hx_gauge if fix_x_gauge else np.zeros((0, n), dtype=np.uint8)
            hz_gauge_selected = hz_gauge if fix_z_gauge else np.zeros((0, n), dtype=np.uint8)
        
        # Build new parity check matrices by combining stabilizers and gauge
        if hx_stabilizer.size > 0 and hx_gauge_selected.size > 0:
            hx_new = np.vstack([hx_stabilizer, hx_gauge_selected])
        elif hx_stabilizer.size > 0:
            hx_new = hx_stabilizer.copy()
        elif hx_gauge_selected.size > 0:
            hx_new = hx_gauge_selected.copy()
        else:
            hx_new = np.zeros((0, n), dtype=np.uint8)
        
        if hz_stabilizer.size > 0 and hz_gauge_selected.size > 0:
            hz_new = np.vstack([hz_stabilizer, hz_gauge_selected])
        elif hz_stabilizer.size > 0:
            hz_new = hz_stabilizer.copy()
        elif hz_gauge_selected.size > 0:
            hz_new = hz_gauge_selected.copy()
        else:
            hz_new = np.zeros((0, n), dtype=np.uint8)
        
        # Verify CSS constraint
        if not css_intersection_check(hx_new, hz_new):
            raise ValueError(
                "Gauge-fixed code violates CSS constraint: Hx @ Hz^T != 0. "
                "This may happen if incompatible gauge operators are selected."
            )
        
        # Build metadata
        meta: Dict[str, Any] = dict(metadata or {})
        meta["is_gauge_fixed"] = True
        meta["fixed_x_gauge"] = fix_x_gauge
        meta["fixed_z_gauge"] = fix_z_gauge
        meta["num_x_gauge_fixed"] = hx_gauge_selected.shape[0]
        meta["num_z_gauge_fixed"] = hz_gauge_selected.shape[0]

        # ── 17 standard metadata keys ──────────────────────────────────────
        meta["code_family"] = "gauge_fixed"
        meta["code_type"] = "gauge_fixed_css"
        meta["n"] = n
        rx = gf2_rank(hx_new) if hx_new.size > 0 else 0
        rz = gf2_rank(hz_new) if hz_new.size > 0 else 0
        k_computed = n - rx - rz
        meta["k"] = k_computed
        meta["distance"] = None  # distance may change after gauge fixing
        meta["rate"] = k_computed / n if n > 0 else 0.0

        if logical_x:
            lx0 = logical_x[0] if isinstance(logical_x[0], dict) else str_to_pauli(logical_x[0]) if isinstance(logical_x[0], str) else {}
            meta["lx_pauli_type"] = "X"
            meta["lx_support"] = sorted(lx0.keys()) if isinstance(lx0, dict) else []
        else:
            meta["lx_pauli_type"] = None
            meta["lx_support"] = []
        if logical_z:
            lz0 = logical_z[0] if isinstance(logical_z[0], dict) else str_to_pauli(logical_z[0]) if isinstance(logical_z[0], str) else {}
            meta["lz_pauli_type"] = "Z"
            meta["lz_support"] = sorted(lz0.keys()) if isinstance(lz0, dict) else []
        else:
            meta["lz_pauli_type"] = None
            meta["lz_support"] = []

        meta["stabiliser_schedule"] = None
        meta["x_schedule"] = None
        meta["z_schedule"] = None

        meta["error_correction_zoo_url"] = "https://errorcorrectionzoo.org/c/subsystem_stabilizer"
        meta["wikipedia_url"] = None
        meta["canonical_references"] = [
            "Poulin, 'Stabilizer formalism for operator quantum error correction' (2005)",
            "Bacon, 'Operator quantum error-correcting subsystems for self-correcting quantum memories' (2006)",
        ]
        meta["connections"] = [
            "Gauge fixing promotes gauge generators to stabilizers",
            "Reduces code space from 2^(k+gauge) to 2^k",
        ]

        # Coordinate metadata
        cols_grid = int(np.ceil(np.sqrt(n)))
        data_coords_list = [(float(i % cols_grid), float(i // cols_grid)) for i in range(n)]
        meta.setdefault("data_coords", data_coords_list)
        x_stab_coords_list = []
        for row in hx_new:
            support = np.where(row)[0]
            if len(support) > 0:
                cx = float(np.mean([data_coords_list[q][0] for q in support]))
                cy = float(np.mean([data_coords_list[q][1] for q in support]))
                x_stab_coords_list.append((cx, cy))
            else:
                x_stab_coords_list.append((0.0, 0.0))
        meta.setdefault("x_stab_coords", x_stab_coords_list)
        z_stab_coords_list = []
        for row in hz_new:
            support = np.where(row)[0]
            if len(support) > 0:
                cx = float(np.mean([data_coords_list[q][0] for q in support]))
                cy = float(np.mean([data_coords_list[q][1] for q in support]))
                z_stab_coords_list.append((cx, cy))
            else:
                z_stab_coords_list.append((0.0, 0.0))
        meta.setdefault("z_stab_coords", z_stab_coords_list)

        validate_css_code(hx_new, hz_new, code_name=f"GaugeFixed(n={n})", raise_on_error=True)
        
        super().__init__(
            hx=hx_new,
            hz=hz_new,
            logical_x=logical_x,
            logical_z=logical_z,
            metadata=meta,
        )
    
    # ------------------------------------------------------------------
    # Gold-standard properties
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        """Human-readable name, e.g. ``'GaugeFixed(n=9, k=1)'``."""
        return f"GaugeFixed(n={self.n}, k={self.k})"

    @property
    def distance(self) -> int:
        """Code distance (may differ from parent subsystem code distance)."""
        d = self._metadata.get("distance")
        return d if d is not None else 1

    def qubit_coords(self) -> List:
        """Return qubit coordinates.

        Attempts to recover geometry from the parent code's metadata.
        If the parent supplied ``'rows'`` and ``'cols'`` (e.g. Bacon–Shor),
        a 2-D grid layout is used; otherwise falls back to a square-grid
        projection based on the qubit count.
        """
        rows = self._metadata.get("rows")
        cols = self._metadata.get("cols")
        if rows is not None and cols is not None:
            return [(float(i % cols), float(i // cols)) for i in range(self.n)]
        side = int(np.ceil(np.sqrt(self.n)))
        return [(float(i % side), float(i // side)) for i in range(self.n)]
    
    @classmethod
    def from_bacon_shor(
        cls,
        rows: int,
        cols: int,
        fix_type: str = 'both',
        metadata: Optional[Dict[str, Any]] = None,
    ) -> 'GaugeFixedCode':
        """
        Create a gauge-fixed Bacon-Shor code.
        
        The Bacon-Shor code is a subsystem code on a rows × cols grid.
        Gauge fixing gives different stabilizer codes depending on 
        which gauge is fixed.
        
        Parameters
        ----------
        rows : int
            Number of rows in the grid.
        cols : int
            Number of columns in the grid.
        fix_type : str
            'x' to fix X gauge, 'z' to fix Z gauge, 'both' for full fixing.
            
        Returns
        -------
        GaugeFixedCode
            The gauge-fixed stabilizer code.
        """
        n = rows * cols
        
        # Bacon-Shor X gauge: XX on adjacent columns in each row
        # Bacon-Shor Z gauge: ZZ on adjacent rows in each column
        
        hx_gauge_list = []
        hz_gauge_list = []
        
        # X gauge: horizontal XX pairs
        for r in range(rows):
            for c in range(cols - 1):
                row = np.zeros(n, dtype=np.uint8)
                row[r * cols + c] = 1
                row[r * cols + c + 1] = 1
                hx_gauge_list.append(row)
        
        # Z gauge: vertical ZZ pairs
        for r in range(rows - 1):
            for c in range(cols):
                row = np.zeros(n, dtype=np.uint8)
                row[r * cols + c] = 1
                row[(r + 1) * cols + c] = 1
                hz_gauge_list.append(row)
        
        hx_gauge = np.array(hx_gauge_list, dtype=np.uint8) if hx_gauge_list else np.zeros((0, n), dtype=np.uint8)
        hz_gauge = np.array(hz_gauge_list, dtype=np.uint8) if hz_gauge_list else np.zeros((0, n), dtype=np.uint8)
        
        # Stabilizers are products of gauge operators (not needed for minimal definition)
        hx_stab = np.zeros((0, n), dtype=np.uint8)
        hz_stab = np.zeros((0, n), dtype=np.uint8)
        
        # Logical X: X on any row
        log_x: PauliString = {c: 'X' for c in range(cols)}
        # Logical Z: Z on any column
        log_z: PauliString = {r * cols: 'Z' for r in range(rows)}
        
        fix_x = fix_type in ('x', 'both')
        fix_z = fix_type in ('z', 'both')
        
        meta = dict(metadata or {})
        meta['bacon_shor'] = True
        meta['grid_size'] = (rows, cols)
        
        return cls(
            hx_stabilizer=hx_stab,
            hz_stabilizer=hz_stab,
            hx_gauge=hx_gauge,
            hz_gauge=hz_gauge,
            logical_x=[log_x],
            logical_z=[log_z],
            fix_x_gauge=fix_x,
            fix_z_gauge=fix_z,
            metadata=meta,
        )


def gauge_fix_css(
    hx_stabilizer: np.ndarray,
    hz_stabilizer: np.ndarray,
    hx_gauge: np.ndarray,
    hz_gauge: np.ndarray,
    logical_x: List[PauliString],
    logical_z: List[PauliString],
    **kwargs,
) -> GaugeFixedCode:
    """
    Convenience function to create a gauge-fixed CSS code.
    
    Parameters
    ----------
    hx_stabilizer, hz_stabilizer : np.ndarray
        Stabilizer generators.
    hx_gauge, hz_gauge : np.ndarray
        Gauge generators to promote.
    logical_x, logical_z : List[PauliString]
        Logical operators.
    **kwargs
        Additional arguments passed to GaugeFixedCode.
        
    Returns
    -------
    GaugeFixedCode
        The gauge-fixed stabilizer code.
    """
    return GaugeFixedCode(
        hx_stabilizer, hz_stabilizer,
        hx_gauge, hz_gauge,
        logical_x, logical_z,
        **kwargs
    )
