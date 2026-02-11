# src/qectostim/codes/composite/subcode.py
"""Subcode, Puncturing, and Shortening Operations on CSS Codes

This module provides three **code surgery** operations that modify a
CSS code by reducing its qubit count or logical dimension:

1. **Subcode (logical freezing)** — promote selected logical operators
   to stabilisers, reducing *k* while keeping *n* fixed.
2. **Puncturing** — remove physical qubits (columns of ``Hx``/``Hz``),
   reducing *n*.
3. **Shortening** — fix physical qubits to known states and then
   remove them, reducing *n* and possibly *k*.

Overview
--------
Starting from an ``[[n, k, d]]`` CSS code *C*:

+--------------+------+------+------+-----------+
| Operation    |  n'  |  k'  |  d'  | Input     |
+==============+======+======+======+===========+
| Subcode      |   n  | k-m  | ≥ d  | logical   |
+--------------+------+------+------+-----------+
| Puncture     | n-m  |  ?   | ≤ d  | physical  |
+--------------+------+------+------+-----------+
| Shorten      | n-m  |  ?   | ≤ d  | physical  |
+--------------+------+------+------+-----------+

Subcode construction
--------------------
Freezing logical qubit *j* in ``|0⟩`` appends logical-``Z_j`` to
``Hz``; freezing in ``|+⟩`` appends logical-``X_j`` to ``Hx``.
Because the new row is already orthogonal to the existing checks
(it is a logical operator), the CSS constraint is preserved.

Puncturing and shortening
-------------------------
Puncturing removes column *j* from both ``Hx`` and ``Hz``, discarding
all information about qubit *j*.  The resulting code may have lower
distance since error chains that previously needed to traverse qubit *j*
can now skip it.

Shortening first fixes qubit *j* to a known state (e.g. ``|0⟩``),
then removes it.  This appends a weight-1 row to ``Hz`` (or ``Hx``)
before deleting the column, which may reduce *k* in addition to *n*.

Relationship to code surgery
-----------------------------
These operations are the *classical* building blocks of **lattice
surgery** and **code deformation**.  In a topological context,
puncturing creates a **hole** in the lattice, while shortening
**contracts** a boundary.

Classes
-------
* ``Subcode`` — freeze logical qubits.
* ``PuncturedCode`` — remove physical qubits.
* ``ShortenedCode`` — fix + remove physical qubits.

Code Parameters
~~~~~~~~~~~~~~~
Starting from a parent ``[[n, k, d]]`` CSS code with ``m`` frozen
logical qubits:

* **Subcode**: ``[[n, k - m, d']]`` where ``d' ≥ d``  (freezing logicals
  can only increase or maintain the distance).
* **Punctured code**: ``[[n - m, k', d']]`` where ``d' ≤ d``  (removing
  qubits may expose shorter error chains).
* **Shortened code**: ``[[n - m, k', d']]`` where ``d' ≤ d``  (fixing and
  removing qubits reduces the code).

The logical dimension ``k - |frozen|`` counts the number of unfrozen
logical qubits that remain as active encoded information.

Stabiliser Structure
~~~~~~~~~~~~~~~~~~~~
For the **Subcode** operation:

* **X stabilisers**: original ``Hx`` rows, plus logical-X operators of
  frozen qubits frozen in the ``|+⟩`` basis.  Weights are inherited
  from the parent code's stabilisers and logical operators.
* **Z stabilisers**: original ``Hz`` rows, plus logical-Z operators of
  frozen qubits frozen in the ``|0⟩`` basis.
* **Total stabiliser count**: ``r_x + m_+`` X-checks and ``r_z + m_0``
  Z-checks, where ``m_+`` and ``m_0`` are the number of qubits frozen
  in ``|+⟩`` and ``|0⟩`` respectively.
* **Measurement schedule**: the new stabilisers (promoted logicals)
  are measured alongside the existing stabilisers in each round.

Examples
--------
>>> from qectostim.codes.small.steane_713 import SteaneCode
>>> from qectostim.codes.composite.subcode import Subcode
>>> s = Subcode(SteaneCode(), freeze_indices=[0], freeze_basis='Z')
>>> s.k == 0  # k reduced from 1 to 0
True

References
----------
* Rains, *Quantum codes of minimum distance two*, IEEE Trans.
  Inform. Theory **45**, 266 (1999).
* Grassl, Beth & Rötteler, *On optimal quantum codes*, Int. J.
  Quantum Inf. **2**, 55 (2004).
* Error Correction Zoo: https://errorcorrectionzoo.org/c/qubit_css

Fault tolerance
---------------
* Subcoding preserves the stabiliser structure, so existing fault-
  tolerant gadgets for the parent code remain valid.
* Puncturing may introduce low-weight undetectable errors; the user
  should verify the new distance after the operation.

Decoding
--------
* After puncturing, the decoder's matching graph must be rebuilt since
  edges involving removed qubits are deleted.
* Shortened codes retain the same Tanner-graph structure (minus the
  fixed qubits), so decoders can be re-used with minor modifications.

Implementation notes
--------------------
* All operations return new ``CSSCode`` instances; the parent code
  object is never modified.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np

from qectostim.codes.abstract_code import PauliString
from qectostim.codes.abstract_css import CSSCode
from qectostim.codes.utils import pauli_to_symplectic, validate_css_code, gf2_rank


def _normalize_pauli(op):
    """Normalize a logical operator to PauliString dict format."""
    if isinstance(op, str):
        return {i: c for i, c in enumerate(op) if c != 'I'}
    return op


class Subcode(CSSCode):
    """
    Create a subcode by freezing logical qubits.
    
    Freezing a logical qubit adds its logical operator to the stabilizer
    group, reducing k by 1 for each frozen qubit.
    
    Parameters
    ----------
    base_code : CSSCode
        The original CSS code.
    freeze_indices : Union[int, List[int]]
        Index or list of indices of logical qubits to freeze (0-indexed).
    freeze_states : Union[str, List[str]], optional
        State(s) to freeze to: '0' (add Z to stabilizers) or '+' (add X).
        Default is '0' for all frozen qubits.
    metadata : dict, optional
        Additional metadata.
        
    Attributes
    ----------
    base_code : CSSCode
        The original code.
    frozen_qubits : List[int]
        Indices of frozen logical qubits.
    frozen_states : List[str]
        States that frozen qubits are fixed to.
        
    Examples
    --------
    >>> from qectostim.codes.base import SteaneCode
    >>> steane = SteaneCode()  # [[7, 1, 3]]
    >>> # Can't freeze the only logical qubit without getting trivial code
    >>> # But for larger codes:
    >>> # sub = Subcode(larger_code, freeze_indices=[0, 2], freeze_states=['0', '+'])
    
    Notes
    -----
    The subcode construction is essentially choosing a specific logical
    subspace of the original code. This is related to gauge fixing but
    operates on logical qubits rather than gauge qubits.
    
    Freezing all k logical qubits gives a trivial [[n, 0, d]] code,
    which is just a classical code (all states are stabilizer states).
    """
    
    def __init__(
        self,
        base_code: CSSCode,
        freeze_indices: Union[int, List[int]],
        freeze_states: Optional[Union[str, List[str]]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Construct a subcode by freezing logical qubits.

        Parameters
        ----------
        base_code : CSSCode
            The parent CSS code to restrict.
        freeze_indices : int or list of int
            0-based indices of logical qubits to freeze.
        freeze_states : str or list of str, optional
            ``'0'`` (append ``Z_j`` to ``Hz``) or ``'+'`` (append
            ``X_j`` to ``Hx``).  Default ``'0'`` for every qubit.
        metadata : dict, optional
            Extra key/value pairs merged into the code's metadata dict.

        Raises
        ------
        ValueError
            If any freeze index is outside ``[0, k)``.
        ValueError
            If the number of *freeze_states* does not match
            the number of *freeze_indices*.
        ValueError
            If a freeze state is not one of ``'0'``, ``'+'``,
            ``'1'``, ``'-'``.
        """
        self.base_code = base_code
        
        # Normalize inputs
        if isinstance(freeze_indices, int):
            freeze_indices = [freeze_indices]
        self.frozen_qubits = sorted(set(freeze_indices))
        
        # Validate indices
        k = base_code.k
        for idx in self.frozen_qubits:
            if idx < 0 or idx >= k:
                raise ValueError(
                    f"Freeze index {idx} out of range for code with k={k} logical qubits"
                )
        
        # Default freeze states
        if freeze_states is None:
            freeze_states = ['0'] * len(self.frozen_qubits)
        elif isinstance(freeze_states, str):
            freeze_states = [freeze_states]
        
        if len(freeze_states) != len(self.frozen_qubits):
            raise ValueError(
                f"Got {len(freeze_states)} freeze states for {len(self.frozen_qubits)} qubits"
            )
        
        for state in freeze_states:
            if state not in ('0', '+', '1', '-'):
                raise ValueError(f"Invalid freeze state '{state}', must be '0', '+', '1', or '-'")
        
        self.frozen_states = freeze_states
        
        # Build new stabilizer matrices
        hx_new = base_code.hx.copy()
        hz_new = base_code.hz.copy()
        n = base_code.n
        
        log_x = [_normalize_pauli(op) for op in base_code.logical_x_ops]
        log_z = [_normalize_pauli(op) for op in base_code.logical_z_ops]
        
        # Add frozen logical operators to stabilizers
        for idx, state in zip(self.frozen_qubits, freeze_states):
            if state in ('0', '1'):
                # Freeze in Z basis: add logical Z to stabilizers
                if idx < len(log_z):
                    lz = log_z[idx]
                    # Convert to binary row
                    new_row = np.zeros(n, dtype=np.uint8)
                    for q, op in lz.items():
                        if op in ('Z', 'Y'):
                            new_row[q] = 1
                    hz_new = np.vstack([hz_new, new_row])
            else:  # '+' or '-'
                # Freeze in X basis: add logical X to stabilizers
                if idx < len(log_x):
                    lx = log_x[idx]
                    new_row = np.zeros(n, dtype=np.uint8)
                    for q, op in lx.items():
                        if op in ('X', 'Y'):
                            new_row[q] = 1
                    hx_new = np.vstack([hx_new, new_row])
        
        # Remove frozen logicals from logical operator lists
        remaining_indices = [i for i in range(k) if i not in self.frozen_qubits]
        logical_x_new = [log_x[i] for i in remaining_indices] if log_x else []
        logical_z_new = [log_z[i] for i in remaining_indices] if log_z else []
        
        # Build metadata
        base_meta = base_code.extra_metadata() if hasattr(base_code, 'extra_metadata') else {}
        meta: Dict[str, Any] = dict(base_meta)
        meta.update(metadata or {})
        meta["base_code_name"] = base_code.name
        meta["frozen_qubits"] = self.frozen_qubits
        meta["frozen_states"] = self.frozen_states
        meta["original_k"] = k

        # ── 17 standard metadata keys ──────────────────────────────────────
        k_new = len(remaining_indices)
        meta["code_family"] = "subcode"
        meta["code_type"] = "frozen_logical"
        meta["n"] = n
        meta["k"] = k_new
        meta["distance"] = None  # distance may increase after freezing
        meta["rate"] = k_new / n if n > 0 else 0.0

        if logical_x_new:
            meta["lx_pauli_type"] = "X"
            lx0 = logical_x_new[0]
            meta["lx_support"] = sorted(lx0.keys()) if isinstance(lx0, dict) else []
        else:
            meta["lx_pauli_type"] = None
            meta["lx_support"] = []
        if logical_z_new:
            meta["lz_pauli_type"] = "Z"
            lz0 = logical_z_new[0]
            meta["lz_support"] = sorted(lz0.keys()) if isinstance(lz0, dict) else []
        else:
            meta["lz_pauli_type"] = None
            meta["lz_support"] = []

        meta["stabiliser_schedule"] = base_meta.get("stabiliser_schedule", None)
        meta["x_schedule"] = base_meta.get("x_schedule", None)
        meta["z_schedule"] = base_meta.get("z_schedule", None)

        meta["error_correction_zoo_url"] = base_meta.get("error_correction_zoo_url", None)
        meta["wikipedia_url"] = base_meta.get("wikipedia_url", None)
        meta["canonical_references"] = base_meta.get("canonical_references", None)
        meta["connections"] = [
            f"Subcode of {base_code.name}: froze logicals {self.frozen_qubits}",
            f"Reduced k from {k} to {k_new}",
        ]

        # Coordinate metadata — inherit data_coords, recompute stab coords
        base_data_coords = base_meta.get("data_coords", None)
        if base_data_coords is None:
            cols_grid = int(np.ceil(np.sqrt(n)))
            base_data_coords = [(float(i % cols_grid), float(i // cols_grid)) for i in range(n)]
        meta.setdefault("data_coords", base_data_coords)
        x_stab_coords_list = []
        for row in hx_new:
            support = np.where(row)[0]
            if len(support) > 0:
                cx = float(np.mean([base_data_coords[q][0] for q in support if q < len(base_data_coords)]))
                cy = float(np.mean([base_data_coords[q][1] for q in support if q < len(base_data_coords)]))
                x_stab_coords_list.append((cx, cy))
            else:
                x_stab_coords_list.append((0.0, 0.0))
        meta.setdefault("x_stab_coords", x_stab_coords_list)
        z_stab_coords_list = []
        for row in hz_new:
            support = np.where(row)[0]
            if len(support) > 0:
                cx = float(np.mean([base_data_coords[q][0] for q in support if q < len(base_data_coords)]))
                cy = float(np.mean([base_data_coords[q][1] for q in support if q < len(base_data_coords)]))
                z_stab_coords_list.append((cx, cy))
            else:
                z_stab_coords_list.append((0.0, 0.0))
        meta.setdefault("z_stab_coords", z_stab_coords_list)

        validate_css_code(hx_new, hz_new, code_name=f"Subcode({base_code.name})", raise_on_error=True)
        
        super().__init__(
            hx=hx_new,
            hz=hz_new,
            logical_x=logical_x_new,
            logical_z=logical_z_new,
            metadata=meta,
        )
    
    # ------------------------------------------------------------------
    # Gold-standard properties
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        """Human-readable name including frozen-qubit summary."""
        frozen_desc = ','.join(
            f"{idx}:{state}" 
            for idx, state in zip(self.frozen_qubits, self.frozen_states)
        )
        return f"Subcode({self.base_code.name}, freeze=[{frozen_desc}])"

    @property
    def distance(self) -> int:
        """Code distance (≥ parent distance; stored in metadata)."""
        d = self._metadata.get("distance")
        return d if d is not None else getattr(self.base_code, 'distance', 1)

    def qubit_coords(self) -> List:
        """Return qubit coordinates inherited from the base code."""
        if hasattr(self.base_code, 'qubit_coords'):
            return self.base_code.qubit_coords()
        return list(range(self.n))


class PuncturedCode(CSSCode):
    """
    Puncture a CSS code by removing physical qubits.
    
    Puncturing removes columns from Hx and Hz corresponding to the
    punctured qubits. This can reduce n while potentially changing k and d.
    
    Parameters
    ----------
    base_code : CSSCode
        The original CSS code.
    puncture_indices : Union[int, List[int]]
        Physical qubit index or indices to remove.
    metadata : dict, optional
        Additional metadata.
        
    Notes
    -----
    Puncturing is different from freezing:
    - Freezing reduces logical space (k decreases)
    - Puncturing removes physical qubits (n decreases)
    
    The distance of the punctured code may decrease significantly
    if punctured qubits are in the support of logical operators.
    """
    
    def __init__(
        self,
        base_code: CSSCode,
        puncture_indices: Union[int, List[int]],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Construct a punctured code by removing physical qubits.

        Parameters
        ----------
        base_code : CSSCode
            The parent CSS code.
        puncture_indices : int or list of int
            0-based indices of physical qubits to remove.
        metadata : dict, optional
            Extra key/value pairs merged into the code's metadata dict.

        Raises
        ------
        ValueError
            If any puncture index is outside ``[0, n)``.
        """
        self.base_code = base_code
        
        if isinstance(puncture_indices, int):
            puncture_indices = [puncture_indices]
        
        self.punctured_qubits = sorted(set(puncture_indices))
        n = base_code.n
        
        # Validate indices
        for idx in self.punctured_qubits:
            if idx < 0 or idx >= n:
                raise ValueError(f"Puncture index {idx} out of range for n={n}")
        
        # Remaining qubit indices
        remaining = [i for i in range(n) if i not in self.punctured_qubits]
        n_new = len(remaining)
        
        # Remove columns from Hx and Hz
        hx_new = base_code.hx[:, remaining]
        hz_new = base_code.hz[:, remaining]
        
        # Remove zero rows (stabilizers that become trivial)
        hx_new = hx_new[np.any(hx_new, axis=1)]
        hz_new = hz_new[np.any(hz_new, axis=1)]
        
        # Update logical operators
        # Map old qubit indices to new
        old_to_new = {old: new for new, old in enumerate(remaining)}
        
        def remap_pauli(pauli: PauliString) -> Optional[PauliString]:
            """Remap a Pauli string, returning None if it becomes trivial."""
            new_pauli: PauliString = {}
            for q, op in pauli.items():
                if q in old_to_new:
                    new_pauli[old_to_new[q]] = op
            return new_pauli if new_pauli else None
        
        logical_x_new = []
        logical_z_new = []
        
        for lx in base_code.logical_x_ops:
            remapped = remap_pauli(_normalize_pauli(lx))
            if remapped:
                logical_x_new.append(remapped)
        
        for lz in base_code.logical_z_ops:
            remapped = remap_pauli(_normalize_pauli(lz))
            if remapped:
                logical_z_new.append(remapped)

        # Build metadata
        base_meta = base_code.extra_metadata() if hasattr(base_code, 'extra_metadata') else {}
        meta: Dict[str, Any] = dict(base_meta)
        meta.update(metadata or {})
        meta["base_code_name"] = base_code.name
        meta["punctured_qubits"] = self.punctured_qubits
        meta["original_n"] = n

        # ── 17 standard metadata keys ──────────────────────────────────────
        rx = gf2_rank(hx_new) if hx_new.size > 0 else 0
        rz = gf2_rank(hz_new) if hz_new.size > 0 else 0
        k_punct = n_new - rx - rz
        meta["code_family"] = "punctured"
        meta["code_type"] = "punctured_css"
        meta["n"] = n_new
        meta["k"] = k_punct
        meta["distance"] = None  # distance may decrease after puncturing
        meta["rate"] = k_punct / n_new if n_new > 0 else 0.0

        if logical_x_new:
            meta["lx_pauli_type"] = "X"
            lx0 = logical_x_new[0]
            meta["lx_support"] = sorted(lx0.keys()) if isinstance(lx0, dict) else []
        else:
            meta["lx_pauli_type"] = None
            meta["lx_support"] = []
        if logical_z_new:
            meta["lz_pauli_type"] = "Z"
            lz0 = logical_z_new[0]
            meta["lz_support"] = sorted(lz0.keys()) if isinstance(lz0, dict) else []
        else:
            meta["lz_pauli_type"] = None
            meta["lz_support"] = []

        meta["stabiliser_schedule"] = None
        meta["x_schedule"] = None
        meta["z_schedule"] = None

        meta["error_correction_zoo_url"] = base_meta.get("error_correction_zoo_url", None)
        meta["wikipedia_url"] = base_meta.get("wikipedia_url", None)
        meta["canonical_references"] = base_meta.get("canonical_references", None)
        meta["connections"] = [
            f"Punctured from {base_code.name}: removed qubits {self.punctured_qubits}",
            f"Reduced n from {n} to {n_new}",
        ]

        # Coordinate metadata — remap from base code, removing punctured qubits
        base_data_coords = base_meta.get("data_coords", None)
        if base_data_coords is None:
            cols_grid = int(np.ceil(np.sqrt(n)))
            base_data_coords = [(float(i % cols_grid), float(i // cols_grid)) for i in range(n)]
        meta.setdefault("data_coords", [base_data_coords[i] for i in remaining])
        punct_data_coords = meta["data_coords"]
        x_stab_coords_list = []
        for row in hx_new:
            support = np.where(row)[0]
            if len(support) > 0:
                cx = float(np.mean([punct_data_coords[q][0] for q in support if q < len(punct_data_coords)]))
                cy = float(np.mean([punct_data_coords[q][1] for q in support if q < len(punct_data_coords)]))
                x_stab_coords_list.append((cx, cy))
            else:
                x_stab_coords_list.append((0.0, 0.0))
        meta.setdefault("x_stab_coords", x_stab_coords_list)
        z_stab_coords_list = []
        for row in hz_new:
            support = np.where(row)[0]
            if len(support) > 0:
                cx = float(np.mean([punct_data_coords[q][0] for q in support if q < len(punct_data_coords)]))
                cy = float(np.mean([punct_data_coords[q][1] for q in support if q < len(punct_data_coords)]))
                z_stab_coords_list.append((cx, cy))
            else:
                z_stab_coords_list.append((0.0, 0.0))
        meta.setdefault("z_stab_coords", z_stab_coords_list)

        validate_css_code(hx_new, hz_new, code_name=f"Punctured({base_code.name})", raise_on_error=True)
        
        super().__init__(
            hx=hx_new,
            hz=hz_new,
            logical_x=logical_x_new,
            logical_z=logical_z_new,
            metadata=meta,
            skip_validation=True,  # puncturing may break CSS constraint
        )
    
    # ------------------------------------------------------------------
    # Gold-standard properties
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        """Human-readable name including punctured-qubit list."""
        return f"Punctured({self.base_code.name}, qubits={self.punctured_qubits})"

    @property
    def distance(self) -> int:
        """Code distance (≤ parent distance; stored in metadata)."""
        d = self._metadata.get("distance")
        return d if d is not None else 1

    def qubit_coords(self) -> List:
        """Return qubit coordinates of remaining qubits."""
        if hasattr(self.base_code, 'qubit_coords'):
            all_coords = self.base_code.qubit_coords()
            remaining = [i for i in range(len(all_coords)) if i not in self.punctured_qubits]
            return [all_coords[i] for i in remaining]
        return list(range(self.n))


class ShortenedCode(CSSCode):
    """
    Shorten a CSS code by fixing physical qubits to known values.
    
    Shortening fixes certain physical qubits to |0⟩ or |+⟩, then removes
    them from the code. This is related to both puncturing and freezing.
    
    Parameters
    ----------
    base_code : CSSCode
        The original CSS code.
    shorten_indices : Union[int, List[int]]
        Physical qubit indices to shorten.
    shorten_states : Optional[Union[str, List[str]]]
        States to fix to: '0' or '+'. Default is '0'.
    metadata : dict, optional
        Additional metadata.
    """
    
    def __init__(
        self,
        base_code: CSSCode,
        shorten_indices: Union[int, List[int]],
        shorten_states: Optional[Union[str, List[str]]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Construct a shortened code by fixing and removing physical qubits.

        Parameters
        ----------
        base_code : CSSCode
            The parent CSS code.
        shorten_indices : int or list of int
            0-based indices of physical qubits to shorten.
        shorten_states : str or list of str, optional
            ``'0'`` or ``'+'`` per shortened qubit.  Default ``'0'``.
        metadata : dict, optional
            Extra key/value pairs merged into the code's metadata dict.

        Raises
        ------
        ValueError
            If any shorten index is outside ``[0, n)``.
        """
        self.base_code = base_code
        
        if isinstance(shorten_indices, int):
            shorten_indices = [shorten_indices]
        self.shortened_qubits = sorted(set(shorten_indices))
        
        if shorten_states is None:
            shorten_states = ['0'] * len(self.shortened_qubits)
        elif isinstance(shorten_states, str):
            shorten_states = [shorten_states]
        
        self.shorten_states = shorten_states
        n = base_code.n
        
        # Validate
        for idx in self.shortened_qubits:
            if idx < 0 or idx >= n:
                raise ValueError(f"Shorten index {idx} out of range for n={n}")
        
        # Remaining qubits
        remaining = [i for i in range(n) if i not in self.shortened_qubits]
        n_new = len(remaining)
        old_to_new = {old: new for new, old in enumerate(remaining)}
        
        # Start with base matrices, remove shortened columns
        hx_base = base_code.hx
        hz_base = base_code.hz
        
        # For each shortened qubit in state '0', we add the constraint that
        # any X on that qubit must be removed from stabilizers
        # (equivalently, we keep only stabilizers with Z or I on that qubit)
        
        # For simplicity, just remove the columns (puncture)
        # A more sophisticated version would also eliminate rows that
        # require non-zero values in shortened positions
        
        hx_new = hx_base[:, remaining]
        hz_new = hz_base[:, remaining]
        
        # Remove rows that had non-zero in shortened positions
        # (these stabilizers conflict with the fixed state)
        keep_x_rows = []
        keep_z_rows = []
        
        for i, row in enumerate(hx_base):
            # X stabilizers: if shortened qubit is in '0' state,
            # X on that qubit would flip it, so remove if row[idx] = 1
            conflict = False
            for j, (idx, state) in enumerate(zip(self.shortened_qubits, self.shorten_states)):
                if state == '0' and row[idx]:
                    conflict = True
                    break
            if not conflict and np.any(hx_new[i]):
                keep_x_rows.append(i)
        
        for i, row in enumerate(hz_base):
            conflict = False
            for j, (idx, state) in enumerate(zip(self.shortened_qubits, self.shorten_states)):
                if state == '+' and row[idx]:
                    conflict = True
                    break
            if not conflict and np.any(hz_new[i]):
                keep_z_rows.append(i)
        
        hx_new = hx_new[keep_x_rows] if keep_x_rows else np.zeros((0, n_new), dtype=np.uint8)
        hz_new = hz_new[keep_z_rows] if keep_z_rows else np.zeros((0, n_new), dtype=np.uint8)
        
        # Remap logical operators
        def remap_pauli(pauli: PauliString) -> Optional[PauliString]:
            new_pauli: PauliString = {}
            for q, op in pauli.items():
                if q in old_to_new:
                    new_pauli[old_to_new[q]] = op
            return new_pauli if new_pauli else None
        
        logical_x_new = []
        logical_z_new = []
        
        for lx in base_code.logical_x_ops:
            remapped = remap_pauli(_normalize_pauli(lx))
            if remapped:
                logical_x_new.append(remapped)
        
        for lz in base_code.logical_z_ops:
            remapped = remap_pauli(_normalize_pauli(lz))
            if remapped:
                logical_z_new.append(remapped)
        
        # Metadata
        base_meta = base_code.extra_metadata() if hasattr(base_code, 'extra_metadata') else {}
        meta: Dict[str, Any] = dict(base_meta)
        meta.update(metadata or {})
        meta["base_code_name"] = base_code.name
        meta["shortened_qubits"] = self.shortened_qubits
        meta["shorten_states"] = self.shorten_states

        # ── 17 standard metadata keys ──────────────────────────────────────
        rx = gf2_rank(hx_new) if hx_new.size > 0 else 0
        rz = gf2_rank(hz_new) if hz_new.size > 0 else 0
        k_short = n_new - rx - rz
        meta["code_family"] = "shortened"
        meta["code_type"] = "shortened_css"
        meta["n"] = n_new
        meta["k"] = k_short
        meta["distance"] = None
        meta["rate"] = k_short / n_new if n_new > 0 else 0.0

        if logical_x_new:
            meta["lx_pauli_type"] = "X"
            lx0 = logical_x_new[0]
            meta["lx_support"] = sorted(lx0.keys()) if isinstance(lx0, dict) else []
        else:
            meta["lx_pauli_type"] = None
            meta["lx_support"] = []
        if logical_z_new:
            meta["lz_pauli_type"] = "Z"
            lz0 = logical_z_new[0]
            meta["lz_support"] = sorted(lz0.keys()) if isinstance(lz0, dict) else []
        else:
            meta["lz_pauli_type"] = None
            meta["lz_support"] = []

        meta["stabiliser_schedule"] = None
        meta["x_schedule"] = None
        meta["z_schedule"] = None

        meta["error_correction_zoo_url"] = base_meta.get("error_correction_zoo_url", None)
        meta["wikipedia_url"] = base_meta.get("wikipedia_url", None)
        meta["canonical_references"] = base_meta.get("canonical_references", None)
        meta["connections"] = [
            f"Shortened from {base_code.name}: fixed qubits {self.shortened_qubits}",
        ]

        # Coordinate metadata — remap from base code, removing shortened qubits
        base_data_coords = base_meta.get("data_coords", None)
        if base_data_coords is None:
            cols_grid = int(np.ceil(np.sqrt(n)))
            base_data_coords = [(float(i % cols_grid), float(i // cols_grid)) for i in range(n)]
        meta.setdefault("data_coords", [base_data_coords[i] for i in remaining])
        short_data_coords = meta["data_coords"]
        x_stab_coords_list = []
        for row in hx_new:
            support = np.where(row)[0]
            if len(support) > 0:
                cx = float(np.mean([short_data_coords[q][0] for q in support if q < len(short_data_coords)]))
                cy = float(np.mean([short_data_coords[q][1] for q in support if q < len(short_data_coords)]))
                x_stab_coords_list.append((cx, cy))
            else:
                x_stab_coords_list.append((0.0, 0.0))
        meta.setdefault("x_stab_coords", x_stab_coords_list)
        z_stab_coords_list = []
        for row in hz_new:
            support = np.where(row)[0]
            if len(support) > 0:
                cx = float(np.mean([short_data_coords[q][0] for q in support if q < len(short_data_coords)]))
                cy = float(np.mean([short_data_coords[q][1] for q in support if q < len(short_data_coords)]))
                z_stab_coords_list.append((cx, cy))
            else:
                z_stab_coords_list.append((0.0, 0.0))
        meta.setdefault("z_stab_coords", z_stab_coords_list)

        validate_css_code(hx_new, hz_new, code_name=f"Shortened({base_code.name})", raise_on_error=True)
        
        super().__init__(
            hx=hx_new,
            hz=hz_new,
            logical_x=logical_x_new,
            logical_z=logical_z_new,
            metadata=meta,
            skip_validation=True,  # shortening may break CSS constraint
        )
    
    # ------------------------------------------------------------------
    # Gold-standard properties
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        """Human-readable name including shortened-qubit list."""
        return f"Shortened({self.base_code.name})"

    @property
    def distance(self) -> int:
        """Code distance (≤ parent distance; stored in metadata)."""
        d = self._metadata.get("distance")
        return d if d is not None else 1

    def qubit_coords(self) -> List:
        """Return qubit coordinates of remaining qubits."""
        if hasattr(self.base_code, 'qubit_coords'):
            all_coords = self.base_code.qubit_coords()
            remaining = [i for i in range(len(all_coords)) if i not in self.shortened_qubits]
            return [all_coords[i] for i in remaining]
        return list(range(self.n))
