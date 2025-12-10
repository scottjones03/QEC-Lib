# src/qectostim/codes/composite/subcode.py
"""
Subcode: Freeze logical qubits by promoting logicals to stabilizers.

Given a [[n, k, d]] code with k logical qubits, a subcode fixes (freezes)
some of the logical qubits by adding their logical operators to the 
stabilizer group.

For CSS codes:
- Freezing logical qubit j in state |0⟩: Add logical Z_j to Hz
- Freezing logical qubit j in state |+⟩: Add logical X_j to Hx

The resulting code is [[n, k-m, d']] where m is the number of frozen qubits
and d' >= d (distance may increase or stay the same).

Common uses:
- Reducing code space for specific applications
- Creating codes with fewer logical qubits from larger codes
- Puncturing/shortening operations
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np

from qectostim.codes.abstract_code import PauliString
from qectostim.codes.abstract_css import CSSCode
from qectostim.codes.utils import pauli_to_symplectic


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
        
        log_x = base_code.logical_x_ops
        log_z = base_code.logical_z_ops
        
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
        
        super().__init__(
            hx=hx_new,
            hz=hz_new,
            logical_x=logical_x_new,
            logical_z=logical_z_new,
            metadata=meta,
        )
    
    @property
    def name(self) -> str:
        frozen_desc = ','.join(
            f"{idx}:{state}" 
            for idx, state in zip(self.frozen_qubits, self.frozen_states)
        )
        return f"Subcode({self.base_code.name}, freeze=[{frozen_desc}])"


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
            remapped = remap_pauli(lx)
            if remapped:
                logical_x_new.append(remapped)
        
        for lz in base_code.logical_z_ops:
            remapped = remap_pauli(lz)
            if remapped:
                logical_z_new.append(remapped)
        
        # Build metadata
        base_meta = base_code.extra_metadata() if hasattr(base_code, 'extra_metadata') else {}
        meta: Dict[str, Any] = dict(base_meta)
        meta.update(metadata or {})
        meta["base_code_name"] = base_code.name
        meta["punctured_qubits"] = self.punctured_qubits
        meta["original_n"] = n
        
        super().__init__(
            hx=hx_new,
            hz=hz_new,
            logical_x=logical_x_new,
            logical_z=logical_z_new,
            metadata=meta,
        )
    
    @property
    def name(self) -> str:
        return f"Punctured({self.base_code.name}, qubits={self.punctured_qubits})"


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
            remapped = remap_pauli(lx)
            if remapped:
                logical_x_new.append(remapped)
        
        for lz in base_code.logical_z_ops:
            remapped = remap_pauli(lz)
            if remapped:
                logical_z_new.append(remapped)
        
        # Metadata
        base_meta = base_code.extra_metadata() if hasattr(base_code, 'extra_metadata') else {}
        meta: Dict[str, Any] = dict(base_meta)
        meta.update(metadata or {})
        meta["base_code_name"] = base_code.name
        meta["shortened_qubits"] = self.shortened_qubits
        meta["shorten_states"] = self.shorten_states
        
        super().__init__(
            hx=hx_new,
            hz=hz_new,
            logical_x=logical_x_new,
            logical_z=logical_z_new,
            metadata=meta,
        )
    
    @property
    def name(self) -> str:
        return f"Shortened({self.base_code.name})"
