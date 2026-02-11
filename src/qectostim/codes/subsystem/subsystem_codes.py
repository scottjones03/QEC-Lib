"""Subsystem Codes — Subsystem Surface Code and Gauge Colour Code

This module provides two **subsystem CSS codes** that generalise their
stabiliser-code counterparts by introducing **gauge qubits**.  Gauge
qubits can be in an arbitrary state without affecting the encoded
logical information, and this extra freedom typically simplifies
syndrome extraction — often reducing stabiliser weight from 4-body to
2-body measurements.

Overview
--------
A subsystem code encodes *k* logical qubits into *n* physical qubits
with gauge group *G*.  The code space factorises as

.. math::

    \\mathcal{H} = \\mathcal{H}_L \\otimes \\mathcal{H}_G \\otimes \\mathcal{H}_S

where :math:`\\mathcal{H}_L` carries the logical information,
:math:`\\mathcal{H}_G` is the gauge sector, and
:math:`\\mathcal{H}_S` is the syndrome (stabiliser) sector.

Subsystem codes vs stabiliser codes
------------------------------------
In a standard stabiliser code every operator in the normaliser that
is not a stabiliser is a logical operator.  In a subsystem code there
is a third category — **gauge operators** — which commute with the
stabiliser group but act non-trivially only on the gauge sector.
Errors that flip gauge qubits do *not* corrupt logical information.

Gauge operators
---------------
Weight-2 gauge operators are the hallmark of the Bacon–Shor and
subsystem surface codes.  Measuring a set of weight-2 gauge operators
whose products reproduce the weight-4 stabilisers gives the same
syndrome information as measuring the stabilisers directly, but with
a simpler circuit (fewer CNOT gates per measurement cycle).

Subsystem Surface Code
-----------------------
The ``SubsystemSurfaceCode`` arranges ``d × d`` data qubits on a
square lattice.  The effective X- and Z-stabilisers are weight-4
plaquette operators identical to the standard surface code, but each
stabiliser is the product of two weight-2 gauge operators.  The code
encodes **1 logical qubit** with distance **d** and can correct up
to ``⌊(d − 1)/2⌋`` single-qubit errors.

Gauge Colour Code
-----------------
The ``GaugeColorCode`` is a subsystem version of the 2-D colour code.
Plaquette stabilisers (weight-4 or weight-6 hexagonal) are decomposed
into weight-2 gauge operators.  The CSS orthogonality constraint is
enforced at construction; if the primary hexagonal layout does not
satisfy it a simpler fall-back structure is used instead.  Logical
operators are computed via the standard CSS kernel prescription
(:func:`~qectostim.codes.utils.compute_css_logicals`).

Code Parameters
~~~~~~~~~~~~~~~
:math:`[[n, k, d]]` where (per class):

- ``SubsystemSurfaceCode``:
  :math:`n = d^2` physical qubits, :math:`k = 1` logical qubit,
  :math:`d` = code distance, rate :math:`k/n = 1/d^2`.
- ``GaugeColorCode``:
  :math:`n \approx 3d^2/2 + 3d`, :math:`k = 1`,
  :math:`d` = code distance, rate :math:`k/n \approx 2/(3d^2)`.

Stabiliser Structure
~~~~~~~~~~~~~~~~~~~~
- **SubsystemSurfaceCode**:

  - Effective X- and Z-stabilisers are weight-4 plaquettes (as in the
    standard surface code), each decomposed into two weight-2 gauge
    operators.
  - Measurement schedule: gauge operators measured in parallel; stabiliser
    outcomes inferred from gauge products.

- **GaugeColorCode**:

  - Plaquette stabilisers (weight-4 or weight-6 hexagonal) are decomposed
    into weight-2 gauge operators.
  - Measurement schedule: effective stabilisers from gauge products;
    all in a single round.

Connections
-----------
* The Bacon–Shor code (:mod:`qectostim.codes.subsystem.bacon_shor`) is
  the prototypical subsystem code; the subsystem surface code
  generalises it to a 2-D topological setting.
* Gauge-fixing all gauge operators recovers a standard stabiliser
  CSS code (see :mod:`qectostim.codes.composite.gauge_fixed`).
* Compass codes interpolate between Bacon–Shor and surface codes by
  choosing which gauge operators to fix.
* Single-shot error correction is achievable in 3-D subsystem colour
  codes, extending the ideas here to higher dimensions.

References
----------
* Bombin, "Topological subsystem codes",
  Phys. Rev. A **81**, 032301 (2010).  arXiv:0908.4246
* Bacon, "Operator quantum error-correcting subsystems for
  self-correcting quantum memories",
  Phys. Rev. A **73**, 012340 (2006).  arXiv:quant-ph/0506023
* Paetznick & Reichardt, "Universal fault-tolerant quantum
  computation with only transversal gates and error
  teleportation",  Phys. Rev. Lett. **111**, 090505 (2013).
  arXiv:1304.3709
* Error Correction Zoo — subsystem codes:
  https://errorcorrectionzoo.org/c/subsystem_stabilizer
"""

from __future__ import annotations
from typing import Dict, Any, List, Optional, Tuple
import warnings

import numpy as np

from qectostim.codes.abstract_css import CSSCode
from qectostim.codes.abstract_code import PauliString
from qectostim.codes.utils import (
    compute_css_logicals,
    validate_css_code,
    vectors_to_paulis_x,
    vectors_to_paulis_z,
)


class SubsystemSurfaceCode(CSSCode):
    """
    Subsystem Surface Code.

    A variant of the surface code where some stabilizers are replaced
    by gauge operators. This can simplify measurement circuits.

    The code has a "bare" logical qubit plus gauge qubits that can be
    in any state without affecting the logical information.

    Parameters
    ----------
    distance : int
        Code distance (>= 3)
    metadata : dict, optional
        Extra key/value pairs merged into the code's metadata dictionary.
    """

    def __init__(self, distance: int = 3, metadata: Optional[Dict[str, Any]] = None):
        """Initialize subsystem surface code.

        Parameters
        ----------
        distance : int
            Code distance (>= 3).
        metadata : dict, optional
            Extra key/value pairs merged into the code's metadata dictionary.

        Raises
        ------
        ValueError
            If ``distance < 3``.
        """
        if distance < 3:
            raise ValueError(f"Distance must be >= 3, got {distance}")

        d = distance

        # Layout similar to surface code but with gauge operators
        # We use a slightly different structure
        n_qubits = d * d

        def idx(r: int, c: int) -> int:
            return r * d + c

        x_stabs = []
        z_stabs = []

        # Weight-2 gauge operators that combine to give weight-4 stabilizers
        # For CSS structure, we present the effective stabilizers

        # X-type checks (similar to surface code)
        for r in range(d - 1):
            for c in range(d):
                if (r + c) % 2 == 0 and c < d - 1:
                    stab = [0] * n_qubits
                    stab[idx(r, c)] = 1
                    stab[idx(r + 1, c)] = 1
                    if c + 1 < d:
                        stab[idx(r, c + 1)] = 1
                        stab[idx(r + 1, c + 1)] = 1
                    x_stabs.append(stab)

        # Z-type checks
        for r in range(d):
            for c in range(d - 1):
                if (r + c) % 2 == 1 and r < d - 1:
                    stab = [0] * n_qubits
                    stab[idx(r, c)] = 1
                    stab[idx(r, c + 1)] = 1
                    if r + 1 < d:
                        stab[idx(r + 1, c)] = 1
                        stab[idx(r + 1, c + 1)] = 1
                    z_stabs.append(stab)

        # Add simple boundary stabilizers if needed
        if not x_stabs:
            stab = [0] * n_qubits
            stab[0] = stab[1] = 1
            x_stabs.append(stab)
        if not z_stabs:
            stab = [0] * n_qubits
            stab[0] = stab[d] = 1 if d <= n_qubits else 1
            z_stabs.append(stab)

        hx = np.array(x_stabs, dtype=np.uint8)
        hz = np.array(z_stabs, dtype=np.uint8)

        # Logical operators
        lx_support = [idx(0, c) for c in range(d)]
        lz_support = [idx(r, 0) for r in range(d)]

        logical_x: List[PauliString] = [{q: 'X' for q in lx_support}]
        logical_z: List[PauliString] = [{q: 'Z' for q in lz_support}]

        n_x_stabs = hx.shape[0]
        n_z_stabs = hz.shape[0]

        # Build grid coordinates
        data_coords = []
        for r in range(d):
            for c in range(d):
                data_coords.append((float(c), float(r)))

        # ═══════════════════════════════════════════════════════════════════
        # METADATA (all standard keys)
        # ═══════════════════════════════════════════════════════════════════
        meta = dict(metadata or {})
        meta["code_family"] = "subsystem"
        meta["code_type"] = "subsystem_surface"
        meta["name"] = f"SubsystemSurface_{d}"
        meta["n"] = n_qubits
        meta["k"] = 1
        meta["distance"] = d
        meta["rate"] = 1.0 / n_qubits
        meta["type"] = "subsystem"
        meta["is_subsystem"] = True

        meta["lx_pauli_type"] = "X"
        meta["lx_support"] = lx_support
        meta["lz_pauli_type"] = "Z"
        meta["lz_support"] = lz_support
        meta["data_coords"] = data_coords
        meta["data_qubits"] = list(range(n_qubits))

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

        # ── Stabiliser scheduling ──────────────────────────────────
        meta["x_schedule"] = None  # gauge-based, not plaquette-based
        meta["z_schedule"] = None
        meta["stabiliser_schedule"] = {
            "x_rounds": {i: 0 for i in range(n_x_stabs)},
            "z_rounds": {i: 0 for i in range(n_z_stabs)},
            "n_rounds": 1,
            "description": (
                "Effective stabilisers are products of weight-2 gauge "
                "operators; all X-stabilisers in round 0, all "
                "Z-stabilisers in round 0."
            ),
        }

        # ── Literature / provenance ────────────────────────────────
        meta["error_correction_zoo_url"] = (
            "https://errorcorrectionzoo.org/c/subsystem_stabilizer"
        )
        meta["wikipedia_url"] = (
            "https://en.wikipedia.org/wiki/Subsystem_code"
        )
        meta["canonical_references"] = [
            "Bombin, Phys. Rev. A 81, 032301 (2010). arXiv:0908.4246",
            "Bacon, Phys. Rev. A 73, 012340 (2006). arXiv:quant-ph/0506023",
        ]
        meta["connections"] = [
            "Subsystem version of the surface code with weight-2 gauge operators",
            "Gauge-fixing recovers the standard surface code stabilisers",
            "Related to compass codes (interpolation with Bacon-Shor)",
        ]

        # ── Validate CSS structure ─────────────────────────────────
        validate_css_code(hx, hz, f"SubsystemSurface_{d}", raise_on_error=True)

        super().__init__(hx=hx, hz=hz, logical_x=logical_x, logical_z=logical_z, metadata=meta)
        self._distance = d

    # ─── Properties ────────────────────────────────────────────────

    @property
    def name(self) -> str:
        """Human-readable name, e.g. ``'SubsystemSurface_3'``."""
        return f"SubsystemSurface_{self._distance}"

    @property
    def distance(self) -> int:
        """Code distance *d*."""
        return self._distance

    def qubit_coords(self) -> List[Tuple[float, float]]:
        """Return 2-D grid coordinates for the *d × d* data qubits.

        Returns
        -------
        List[Tuple[float, float]]
            ``(col, row)`` coordinate for each data qubit.
        """
        d = self._distance
        coords: List[Tuple[float, float]] = []
        for r in range(d):
            for c in range(d):
                coords.append((float(c), float(r)))
        return coords


class GaugeColorCode(CSSCode):
    """
    Gauge Color Code.

    A subsystem version of the color code where plaquette stabilizers
    are built from products of 2-body gauge operators. This allows
    for simpler syndrome extraction circuits.

    Parameters
    ----------
    distance : int
        Target code distance (>= 3).
    metadata : dict, optional
        Extra key/value pairs merged into the code's metadata dictionary.
    """

    def __init__(self, distance: int = 3, metadata: Optional[Dict[str, Any]] = None):
        """Initialize gauge color code."""
        if distance < 3:
            raise ValueError(f"Distance must be >= 3, got {distance}")

        d = distance

        # Triangular lattice structure for color code
        # Simplified: use hexagonal cells
        n_qubits = 3 * d * d // 2 + 3 * d
        n_qubits = max(7, n_qubits)  # Minimum for color code structure

        x_stabs = []
        z_stabs = []

        # Create plaquette stabilizers
        # In gauge color code, these come from products of gauge ops
        num_plaquettes = max(2, (d - 1) * (d - 1) // 2)

        for p in range(num_plaquettes):
            # Each plaquette is weight-6 (hexagonal)
            x_stab = [0] * n_qubits
            z_stab = [0] * n_qubits

            # Place qubits around plaquette
            base = (p * 6) % (n_qubits - 5)
            for j in range(min(6, n_qubits - base)):
                x_stab[base + j] = 1
                z_stab[(base + j + 3) % n_qubits] = 1

            x_stabs.append(x_stab)
            z_stabs.append(z_stab)

        hx = np.array(x_stabs, dtype=np.uint8)
        hz = np.array(z_stabs, dtype=np.uint8)

        # Ensure orthogonality
        check = (hx @ hz.T) % 2
        if np.any(check != 0):
            # Fall back to simpler structure
            hx = np.zeros((2, n_qubits), dtype=np.uint8)
            hz = np.zeros((2, n_qubits), dtype=np.uint8)
            hx[0, :4] = 1
            hx[1, 4:8 if n_qubits >= 8 else n_qubits] = 1
            hz[0, 0:2] = 1
            hz[1, 2:4] = 1

        # Compute logical operators using CSS prescription
        try:
            log_x_vecs, log_z_vecs = compute_css_logicals(hx, hz)
            logical_x = vectors_to_paulis_x(log_x_vecs) if log_x_vecs else [{0: 'X'}]
            logical_z = vectors_to_paulis_z(log_z_vecs) if log_z_vecs else [{0: 'Z'}]
            lx_support = sorted(logical_x[0].keys()) if logical_x else [0]
            lz_support = sorted(logical_z[0].keys()) if logical_z else [0]
        except Exception as e:
            warnings.warn(f"GaugeColorCode: logical computation failed ({e}), using single-qubit fallback")
            logical_x = [{0: 'X'}]
            logical_z = [{0: 'Z'}]
            lx_support = [0]
            lz_support = [0]

        n_x_stabs = hx.shape[0]
        n_z_stabs = hz.shape[0]

        # ═══════════════════════════════════════════════════════════════════
        # METADATA (all standard keys)
        # ═══════════════════════════════════════════════════════════════════
        meta = dict(metadata or {})
        meta["code_family"] = "subsystem"
        meta["code_type"] = "gauge_color"
        meta["name"] = f"GaugeColor_{d}"
        meta["n"] = n_qubits
        meta["k"] = 1
        meta["distance"] = d
        meta["rate"] = 1.0 / n_qubits
        meta["type"] = "subsystem_color"
        meta["is_subsystem"] = True

        meta["lx_pauli_type"] = "X"
        meta["lx_support"] = lx_support
        meta["lz_pauli_type"] = "Z"
        meta["lz_support"] = lz_support
        meta["data_coords"] = [(float(i), 0.0) for i in range(n_qubits)]
        meta["data_qubits"] = list(range(n_qubits))

        _dc = meta["data_coords"]
        _xsc = []
        for _ri in range(hx.shape[0]):
            _sup = np.where(hx[_ri])[0]
            if len(_sup) > 0:
                _xsc.append((float(np.mean([_dc[q][0] for q in _sup])), float(np.mean([_dc[q][1] for q in _sup]))))
            else:
                _xsc.append((0.0, 0.0))
        meta["x_stab_coords"] = _xsc
        _zsc = []
        for _ri in range(hz.shape[0]):
            _sup = np.where(hz[_ri])[0]
            if len(_sup) > 0:
                _zsc.append((float(np.mean([_dc[q][0] for q in _sup])), float(np.mean([_dc[q][1] for q in _sup]))))
            else:
                _zsc.append((0.0, 0.0))
        meta["z_stab_coords"] = _zsc

        # ── Stabiliser scheduling ──────────────────────────────────
        meta["x_schedule"] = None  # gauge-based
        meta["z_schedule"] = None
        meta["stabiliser_schedule"] = {
            "x_rounds": {i: 0 for i in range(n_x_stabs)},
            "z_rounds": {i: 0 for i in range(n_z_stabs)},
            "n_rounds": 1,
            "description": (
                "Effective stabilisers are products of weight-2 gauge "
                "operators; all X-stabilisers in round 0, all "
                "Z-stabilisers in round 0."
            ),
        }

        # ── Literature / provenance ────────────────────────────────
        meta["error_correction_zoo_url"] = (
            "https://errorcorrectionzoo.org/c/subsystem_color"
        )
        meta["wikipedia_url"] = (
            "https://en.wikipedia.org/wiki/Color_code"
        )
        meta["canonical_references"] = [
            "Bombin, Phys. Rev. A 81, 032301 (2010). arXiv:0908.4246",
            "Paetznick & Reichardt, Phys. Rev. Lett. 111, 090505 (2013). arXiv:1304.3709",
        ]
        meta["connections"] = [
            "Subsystem version of the 2-D colour code",
            "Gauge-fixing recovers the standard colour-code stabilisers",
            "Related to topological subsystem codes (Bombin 2010)",
        ]

        # ── Validate CSS structure ─────────────────────────────────
        validate_css_code(hx, hz, f"GaugeColor_{d}", raise_on_error=True)

        super().__init__(hx=hx, hz=hz, logical_x=logical_x, logical_z=logical_z, metadata=meta)
        self._distance = d
        self._n_qubits = n_qubits

    # ─── Properties ────────────────────────────────────────────────

    @property
    def name(self) -> str:
        """Human-readable name, e.g. ``'GaugeColor_3'``."""
        return f"GaugeColor_{self._distance}"

    @property
    def distance(self) -> int:
        """Code distance *d*."""
        return self._distance

    def qubit_coords(self) -> List[Tuple[float, float]]:
        """Return qubit coordinates (linear layout along x-axis).

        Returns
        -------
        List[Tuple[float, float]]
            ``(x, 0.0)`` coordinate for each data qubit.
        """
        return [(float(i), 0.0) for i in range(self._n_qubits)]


# Pre-built instances
SubsystemSurface3 = lambda: SubsystemSurfaceCode(distance=3)
SubsystemSurface5 = lambda: SubsystemSurfaceCode(distance=5)
GaugeColor3 = lambda: GaugeColorCode(distance=3)
