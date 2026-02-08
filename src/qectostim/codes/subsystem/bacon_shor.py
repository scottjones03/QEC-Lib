"""Bacon–Shor Subsystem Code

The Bacon–Shor code is a **subsystem code** on an *m × n* grid of
physical qubits encoding **1 logical qubit** with distance
``d = min(m, n)``.

Key advantage
-------------
All syndrome information can be extracted with **weight-2 gauge
operators** (nearest-neighbour XX or ZZ), unlike surface codes
which require weight-4 stabiliser measurements.

Construction
------------
* Physical qubits are arranged on an ``m × n`` grid
  (``n_phys = m · n``).
* **Gauge operators** (weight 2):

  - X-type: ``XX`` on each horizontal pair in every row
  - Z-type: ``ZZ`` on each vertical pair in every column

* **Stabilisers** are *products* of gauge operators:

  - X-stabiliser for row pair ``(r, r+1)``: X on all qubits in
    rows *r* and *r+1*  → weight 2n
  - Z-stabiliser for col pair ``(c, c+1)``: Z on all qubits in
    cols *c* and *c+1*  → weight 2m

Code parameters
---------------
* **n** = m · n  physical qubits
* **k** = 1      logical qubit
* **d** = min(m, n)
* **Rate** R = 1 / (m · n)
* **Gauge qubits**: ``m · n − 1 − (m − 1) − (n − 1)`` = ``(m−1)(n−1)``

Logical operators
-----------------
* X̄ = X on every qubit in any single row   (weight n)
* Z̄ = Z on every qubit in any single column (weight m)

Connections
-----------
* **Shor code** [[9,1,3]] is the 3 × 3 Bacon–Shor code.
* **Subsystem surface codes** generalise this idea to 2-D topologies.
* **Compass codes** interpolate between Bacon–Shor and surface codes.

References
----------
* Bacon, "Operator quantum error-correcting subsystems for
  self-correcting quantum memories", Phys. Rev. A **73**, 012340
  (2006).  arXiv:quant-ph/0506023
* Aliferis & Cross, "Subsystem fault tolerance with the Bacon–Shor
  code", Phys. Rev. Lett. **98**, 220502 (2007).  arXiv:quant-ph/0610063
* Error Correction Zoo: https://errorcorrectionzoo.org/c/bacon_shor_classical
"""

from __future__ import annotations
from typing import Dict, Any, List, Optional, Tuple

import numpy as np

from qectostim.codes.abstract_css import CSSCode
from qectostim.codes.abstract_code import PauliString
from qectostim.codes.complexes.css_complex import CSSChainComplex3
from qectostim.codes.utils import validate_css_code

Coord2D = Tuple[float, float]


class BaconShorCode(CSSCode):
    """Bacon–Shor subsystem code on an *m × n* grid.

    Encodes 1 logical qubit in ``m × n`` physical qubits with distance
    ``min(m, n)``.  Syndrome extraction uses only **weight-2** gauge
    operators, simplifying hardware requirements.

    Parameters
    ----------
    m : int
        Number of rows (≥ 2).
    n : int
        Number of columns (≥ 2).
    metadata : dict, optional
        Extra key/value pairs merged into the code's metadata dictionary.

    Attributes
    ----------
    n : int
        Number of physical qubits (``m × n``).
    k : int
        Number of logical qubits (1).
    distance : int
        Code distance (``min(m, n)``).
    hx : np.ndarray
        X-stabiliser parity-check matrix, shape ``(m-1, m*n)``.
    hz : np.ndarray
        Z-stabiliser parity-check matrix, shape ``(n-1, m*n)``.

    Examples
    --------
    >>> code = BaconShorCode(3, 3)
    >>> code.n, code.k, code.distance
    (9, 1, 3)
    >>> x_gauges, z_gauges = code.gauge_operators()
    >>> len(x_gauges), len(z_gauges)  # weight-2 gauge ops
    (6, 6)

    Notes
    -----
    The 3 × 3 Bacon–Shor code is equivalent to Shor's [[9,1,3]] code
    up to a local Clifford rotation.  The gauge structure allows single-
    shot error correction with only 2-body measurements.

    See Also
    --------
    ShorCode91 : The 3×3 Bacon–Shor code in CSS form.
    """

    def __init__(self, m: int = 3, n: int = 3, metadata: Optional[Dict[str, Any]] = None):
        """Initialise the Bacon–Shor code on an *m × n* grid.

        Builds row-pair X-stabilisers, column-pair Z-stabilisers,
        weight-2 gauge operators, logical operators, a chain complex,
        and all standard metadata fields.

        Parameters
        ----------
        m : int, optional
            Number of rows (default 3, must be ≥ 2).
        n : int, optional
            Number of columns (default 3, must be ≥ 2).
        metadata : dict, optional
            Extra metadata merged into the code's metadata dictionary.
        """
        if m < 2 or n < 2:
            raise ValueError(f"Grid must be at least 2×2, got {m}×{n}")
        
        n_qubits = m * n
        
        def qubit_idx(row: int, col: int) -> int:
            return row * n + col
        
        # Gauge operators (weight-2):
        # X-type: XX on horizontal pairs (same row)
        # Z-type: ZZ on vertical pairs (same column)
        
        # Stabilizers are products of gauge operators:
        # X stabilizers: product of XX gauges in each row (gives all-X on row)
        # Z stabilizers: product of ZZ gauges in each column (gives all-Z on column)
        
        # For CSS framework, we use the stabilizers (not gauges)
        # X-stabilizer: All X on each row (m-1 independent)
        # Z-stabilizer: All Z on each column (n-1 independent)
        
        # Actually, for Bacon-Shor the stabilizers are:
        # - Product of all X in row i times all X in row j (i<j)
        # - Product of all Z in col i times all Z in col j (i<j)
        # This gives row-pair and column-pair stabilizers
        
        # Simpler: X-type on pairs of rows, Z-type on pairs of columns
        x_stabs = []
        z_stabs = []
        
        # X stabilizers: for each pair of adjacent rows
        for r in range(m - 1):
            stab = [0] * n_qubits
            for c in range(n):
                stab[qubit_idx(r, c)] = 1
                stab[qubit_idx(r + 1, c)] = 1
            x_stabs.append(stab)
        
        # Z stabilizers: for each pair of adjacent columns
        for c in range(n - 1):
            stab = [0] * n_qubits
            for r in range(m):
                stab[qubit_idx(r, c)] = 1
                stab[qubit_idx(r, c + 1)] = 1
            z_stabs.append(stab)
        
        hx = np.array(x_stabs, dtype=np.uint8) if x_stabs else np.zeros((0, n_qubits), dtype=np.uint8)
        hz = np.array(z_stabs, dtype=np.uint8) if z_stabs else np.zeros((0, n_qubits), dtype=np.uint8)
        
        # Gauge operators (store in metadata)
        x_gauges = []  # XX on horizontal pairs
        z_gauges = []  # ZZ on vertical pairs
        
        for r in range(m):
            for c in range(n - 1):
                x_gauges.append((qubit_idx(r, c), qubit_idx(r, c + 1)))
        
        for c in range(n):
            for r in range(m - 1):
                z_gauges.append((qubit_idx(r, c), qubit_idx(r + 1, c)))
        
        # Logical operators
        # Logical X: all X on any row
        lx_support = [qubit_idx(0, c) for c in range(n)]
        
        # Logical Z: all Z on any column
        lz_support = [qubit_idx(r, 0) for r in range(m)]
        
        logical_x: List[PauliString] = [{q: 'X' for q in lx_support}]
        logical_z: List[PauliString] = [{q: 'Z' for q in lz_support}]
        
        distance = min(m, n)
        
        meta = dict(metadata or {})
        meta["name"] = f"BaconShor_{m}x{n}"
        meta["n"] = n_qubits
        meta["k"] = 1
        meta["distance"] = distance
        meta["m"] = m
        meta["grid_n"] = n
        meta["is_subsystem"] = True
        meta["x_gauges"] = x_gauges
        meta["z_gauges"] = z_gauges
        
        # Grid coordinates
        coords = {}
        for r in range(m):
            for c in range(n):
                coords[qubit_idx(r, c)] = (c, m - 1 - r)
        meta["data_coords"] = [coords[i] for i in range(n_qubits)]
        
        super().__init__(hx=hx, hz=hz, logical_x=logical_x, logical_z=logical_z, metadata=meta)
    
    def gauge_operators(self) -> Tuple[List[PauliString], List[PauliString]]:
        """Return the gauge operators (weight-2).
        
        Returns
        -------
        Tuple[List[PauliString], List[PauliString]]
            (X-type gauges, Z-type gauges)
        """
        x_gauges = self._metadata.get("x_gauges", [])
        z_gauges = self._metadata.get("z_gauges", [])
        
        x_ops = []
        for q1, q2 in x_gauges:
            op = {q1: 'X', q2: 'X'}
            x_ops.append(op)
        
        z_ops = []
        for q1, q2 in z_gauges:
            op = {q1: 'Z', q2: 'Z'}
            z_ops.append(op)
        
        return x_ops, z_ops


# Pre-built instances
BaconShor3x3 = lambda: BaconShorCode(3, 3)
BaconShor4x4 = lambda: BaconShorCode(4, 4)
BaconShor5x5 = lambda: BaconShorCode(5, 5)
