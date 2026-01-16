# src/qectostim/experiments/stabilizer_rounds/context.py
"""
Detector context for tracking measurements and emitting detectors.

The DetectorContext is the central state object that tracks measurement indices,
stabilizer history, and observable transformations throughout circuit construction.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import stim


@dataclass
class DetectorContext:
    """
    Track measurement indices and emit detectors for fault-tolerant circuits.
    
    This context manages the state needed to correctly emit DETECTOR instructions
    in Stim circuits. It tracks:
    
    1. Measurement indices: Global counter of measurements in the circuit
    2. Stabilizer history: Last measurement index for each stabilizer
    3. Time coordinates: For visualizing space-time error diagrams
    4. Observable tracking: Measurement indices contributing to logical observables
    
    The context is designed to be passed between different phases of circuit
    construction (preparation, rounds, gadgets, final measurement) to maintain
    consistency of detector references.
    
    Attributes
    ----------
    measurement_index : int
        Current total measurement count (next measurement gets this index).
    last_stabilizer_meas : dict
        Maps (block_name, stab_type, stab_idx) -> last measurement index.
    current_time : float
        Current time coordinate for detector emission.
    time_step : float
        Amount to advance time per round.
    observable_measurements : dict
        Maps observable_idx -> list of measurement indices contributing to it.
    observable_transforms : dict
        Maps observable_idx -> dict of Pauli transformations applied.
    stabilizer_initialized : set
        Set of (block_name, stab_type, stab_idx) for stabilizers with baseline.
    """
    
    measurement_index: int = 0
    last_stabilizer_meas: Dict[Tuple[str, str, int], int] = field(default_factory=dict)
    current_time: float = 0.0
    time_step: float = 1.0
    observable_measurements: Dict[int, List[int]] = field(default_factory=dict)
    observable_transforms: Dict[int, Dict[str, str]] = field(default_factory=dict)
    stabilizer_initialized: Set[Tuple[str, str, int]] = field(default_factory=set)
    
    # Transform tracking for hierarchical concatenated codes
    # After CZ(block_A, block_B): X_A → X_A ⊗ Z_B, X_B → X_B ⊗ Z_A
    # After CNOT(ctrl, tgt): X_ctrl → X_ctrl ⊗ X_tgt, Z_tgt → Z_ctrl ⊗ Z_tgt
    # Maps (block_name, stab_type, stab_idx) → list of partner measurement indices
    pending_x_transforms: Dict[Tuple[str, int], List[int]] = field(default_factory=dict)
    pending_z_transforms: Dict[Tuple[str, int], List[int]] = field(default_factory=dict)
    
    # Inner stabilizer measurement history for hierarchical decoder
    # Key: (block_name, stab_type, stab_idx) -> List[meas_idx] in chronological order
    # This enables the decoder to access ALL inner syndrome measurements, not just the last
    stabilizer_measurement_history: Dict[Tuple[str, str, int], List[int]] = field(default_factory=dict)
    
    # Round counter for tracking which round measurements belong to
    current_round: int = 0
    
    def add_measurement(self, count: int = 1) -> int:
        """
        Add measurements and return the starting index.
        
        Parameters
        ----------
        count : int
            Number of measurements to add.
            
        Returns
        -------
        int
            Starting index of the new measurements.
        """
        start = self.measurement_index
        self.measurement_index += count
        return start
    
    def record_stabilizer_measurement(
        self,
        block_name: str,
        stab_type: str,
        stab_idx: int,
        meas_idx: int,
    ) -> Optional[int]:
        """
        Record a stabilizer measurement and return the previous index.
        
        Parameters
        ----------
        block_name : str
            Name of the code block (for multi-code gadgets).
        stab_type : str
            Type of stabilizer ("x", "z", or "xyz").
        stab_idx : int
            Index of the stabilizer.
        meas_idx : int
            Measurement index for this round.
            
        Returns
        -------
        Optional[int]
            Previous measurement index, or None if this is the first round.
        """
        key = (block_name, stab_type, stab_idx)
        prev = self.last_stabilizer_meas.get(key)
        self.last_stabilizer_meas[key] = meas_idx
        self.stabilizer_initialized.add(key)
        
        # Also record in history for hierarchical decoder
        if key not in self.stabilizer_measurement_history:
            self.stabilizer_measurement_history[key] = []
        self.stabilizer_measurement_history[key].append(meas_idx)
        
        return prev
    
    def advance_time(self, delta: Optional[float] = None) -> None:
        """Advance the time coordinate."""
        self.current_time += delta if delta is not None else self.time_step
    
    def advance_round(self) -> None:
        """Advance the round counter."""
        self.current_round += 1
    
    def get_inner_measurement_history_by_block(self) -> Dict[str, Dict[str, List[int]]]:
        """
        Get inner stabilizer measurement history organized by block.
        
        Returns
        -------
        Dict[str, Dict[str, List[int]]]
            block_name -> {
                'x_anc': [meas_idx, ...],  # All X ancilla measurements in order
                'z_anc': [meas_idx, ...],  # All Z ancilla measurements in order
            }
        """
        result: Dict[str, Dict[str, List[int]]] = {}
        
        for (block_name, stab_type, stab_idx), meas_list in self.stabilizer_measurement_history.items():
            if block_name not in result:
                result[block_name] = {'x_anc': [], 'z_anc': []}
            
            key = 'x_anc' if stab_type == 'x' else 'z_anc'
            # Add all measurements for this stabilizer
            result[block_name][key].extend(meas_list)
        
        # Sort measurements within each category to maintain chronological order
        for block_name in result:
            result[block_name]['x_anc'] = sorted(result[block_name]['x_anc'])
            result[block_name]['z_anc'] = sorted(result[block_name]['z_anc'])
        
        return result
    
    def add_observable_measurement(
        self,
        observable_idx: int,
        meas_indices: List[int],
    ) -> None:
        """
        Add measurement indices to an observable's accumulator.
        
        For observables that span multiple phases (e.g., measurements
        before and after a gadget), this accumulates all contributing
        measurement indices.
        """
        if observable_idx not in self.observable_measurements:
            self.observable_measurements[observable_idx] = []
        self.observable_measurements[observable_idx].extend(meas_indices)
    
    def record_observable_transform(
        self,
        observable_idx: int,
        transform: Dict[str, str],
    ) -> None:
        """
        Record how a gate transforms the observable.
        
        For tracking how logical X/Z change through the circuit.
        E.g., after Hadamard: {'X': 'Z', 'Z': 'X'}
        """
        if observable_idx not in self.observable_transforms:
            self.observable_transforms[observable_idx] = {'X': 'X', 'Z': 'Z', 'Y': 'Y'}
        
        # Compose transformations
        current = self.observable_transforms[observable_idx]
        new_transform = {}
        for pauli, result in current.items():
            # Strip sign for lookup, preserve for result
            lookup = result.lstrip('-')
            sign = '-' if result.startswith('-') else ''
            if lookup in transform:
                new_result = transform[lookup]
                # Compose signs
                new_sign = '-' if (sign == '-') != new_result.startswith('-') else ''
                new_transform[pauli] = new_sign + new_result.lstrip('-')
            else:
                new_transform[pauli] = result
        
        self.observable_transforms[observable_idx] = new_transform
    
    def get_transformed_basis(self, observable_idx: int, original_basis: str) -> str:
        """Get the current basis of an observable after all transformations."""
        if observable_idx not in self.observable_transforms:
            return original_basis
        transform = self.observable_transforms[observable_idx]
        result = transform.get(original_basis, original_basis)
        return result.lstrip('-')  # Return just the basis, not the sign
    
    def emit_detector(
        self,
        circuit: stim.Circuit,
        meas_indices: List[int],
        coord: Tuple[float, ...],
    ) -> None:
        """
        Emit a DETECTOR instruction.
        
        Parameters
        ----------
        circuit : stim.Circuit
            Target circuit.
        meas_indices : List[int]
            Absolute measurement indices to include.
        coord : Tuple[float, ...]
            Detector coordinates (x, y, t) or (x, y, t, basis).
        """
        if not meas_indices:
            return
        
        lookbacks = [idx - self.measurement_index for idx in meas_indices]
        targets = [stim.target_rec(lb) for lb in lookbacks]
        circuit.append("DETECTOR", targets, list(coord))
    
    def emit_observable(
        self,
        circuit: stim.Circuit,
        observable_idx: int = 0,
    ) -> None:
        """
        Emit OBSERVABLE_INCLUDE for accumulated measurements.
        
        Parameters
        ----------
        circuit : stim.Circuit
            Target circuit.
        observable_idx : int
            Which logical observable.
        """
        meas_indices = self.observable_measurements.get(observable_idx, [])
        if not meas_indices:
            return
        
        lookbacks = [idx - self.measurement_index for idx in meas_indices]
        targets = [stim.target_rec(lb) for lb in lookbacks]
        circuit.append("OBSERVABLE_INCLUDE", targets, observable_idx)
    
    def clone(self) -> "DetectorContext":
        """Create a copy of the context."""
        return DetectorContext(
            measurement_index=self.measurement_index,
            last_stabilizer_meas=dict(self.last_stabilizer_meas),
            current_time=self.current_time,
            time_step=self.time_step,
            observable_measurements={k: list(v) for k, v in self.observable_measurements.items()},
            observable_transforms={k: dict(v) for k, v in self.observable_transforms.items()},
            stabilizer_initialized=set(self.stabilizer_initialized),
            pending_x_transforms={k: list(v) for k, v in self.pending_x_transforms.items()},
            pending_z_transforms={k: list(v) for k, v in self.pending_z_transforms.items()},
        )
    
    def update_for_gate(self, gate_name: str) -> None:
        """
        Update observable tracking for a gate transformation.
        
        This records how the logical observables transform through
        the gate, which affects how final measurements should be
        interpreted for the logical observable.
        
        Parameters
        ----------
        gate_name : str
            Name of the gate (H, S, T, CNOT, CZ, etc.)
        """
        # Gate transformation rules for Pauli observables
        gate_transforms = {
            # Hadamard: swaps X and Z
            "H": {"X": "Z", "Y": "-Y", "Z": "X"},
            # S gate: X -> Y, Y -> -X, Z -> Z
            "S": {"X": "Y", "Y": "-X", "Z": "Z"},
            "S_DAG": {"X": "-Y", "Y": "X", "Z": "Z"},
            # T gate: X -> (X+Y)/√2, but for logical tracking we use X -> X
            # (exact phase not needed for detector matching)
            "T": {"X": "X", "Y": "Y", "Z": "Z"},
            "T_DAG": {"X": "X", "Y": "Y", "Z": "Z"},
            # Pauli gates: just apply phases
            "X": {"X": "X", "Y": "-Y", "Z": "-Z"},
            "Y": {"X": "-X", "Y": "Y", "Z": "-Z"},
            "Z": {"X": "-X", "Y": "-Y", "Z": "Z"},
            # CNOT: X_c -> X_c X_t, Z_t -> Z_c Z_t
            # (Handled specially for two-qubit)
            "CNOT": {"X": "X", "Y": "Y", "Z": "Z"},  # Placeholder
            "CX": {"X": "X", "Y": "Y", "Z": "Z"},
            "CZ": {"X": "X", "Y": "Y", "Z": "Z"},  # Symmetric
        }
        
        transform = gate_transforms.get(gate_name.upper(), {"X": "X", "Y": "Y", "Z": "Z"})
        
        # Apply to observable 0 (can extend to multiple observables)
        self.record_observable_transform(0, transform)
    
    def clear_stabilizer_history(
        self,
        block_name: Optional[str] = None,
        swap_xz: bool = False,
    ) -> None:
        """
        Clear stabilizer measurement history at gate boundaries.
        
        This is necessary when a gate changes the interpretation of stabilizers,
        such as a Hadamard which swaps X and Z stabilizers. After clearing,
        the next stabilizer round will establish a new baseline rather than
        creating invalid time-like detectors.
        
        Parameters
        ----------
        block_name : str, optional
            If provided, only clear history for this block.
            If None, clear all history.
        swap_xz : bool
            If True, swap X and Z keys in the history instead of clearing.
            This preserves continuity when X stabilizers become Z stabilizers.
        """
        if swap_xz:
            # Swap X and Z stabilizer entries instead of clearing
            new_meas = {}
            new_init = set()
            for key, val in self.last_stabilizer_meas.items():
                bname, stab_type, stab_idx = key
                if block_name is None or bname == block_name:
                    # Swap x <-> z
                    new_type = "z" if stab_type == "x" else "x" if stab_type == "z" else stab_type
                    new_key = (bname, new_type, stab_idx)
                    new_meas[new_key] = val
                else:
                    new_meas[key] = val
            
            for key in self.stabilizer_initialized:
                bname, stab_type, stab_idx = key
                if block_name is None or bname == block_name:
                    new_type = "z" if stab_type == "x" else "x" if stab_type == "z" else stab_type
                    new_init.add((bname, new_type, stab_idx))
                else:
                    new_init.add(key)
            
            self.last_stabilizer_meas = new_meas
            self.stabilizer_initialized = new_init
        else:
            if block_name is None:
                self.last_stabilizer_meas.clear()
                self.stabilizer_initialized.clear()
            else:
                keys_to_remove = [k for k in self.last_stabilizer_meas if k[0] == block_name]
                for k in keys_to_remove:
                    del self.last_stabilizer_meas[k]
                self.stabilizer_initialized = {
                    k for k in self.stabilizer_initialized if k[0] != block_name
                }
    
    # =========================================================================
    # Transform Tracking for Hierarchical Concatenated Codes
    # =========================================================================
    
    def record_cz_transform(
        self,
        block_a: str,
        block_b: str,
        n_x_stabs_a: int,
        n_x_stabs_b: int,
    ) -> None:
        """
        Record stabilizer transforms after a logical CZ gate.
        
        After CZ(A, B):
        - X stabilizers on A transform: X_A → X_A ⊗ Z_B
        - X stabilizers on B transform: X_B → X_B ⊗ Z_A
        - Z stabilizers are unchanged
        
        This means X detector on A needs B's Z measurements, and vice versa.
        
        Parameters
        ----------
        block_a : str
            Name of first block.
        block_b : str
            Name of second block.
        n_x_stabs_a : int
            Number of X stabilizers in block A.
        n_x_stabs_b : int
            Number of X stabilizers in block B.
        """
        # Get last Z measurements from both blocks (all Z stabilizers)
        z_meas_a = self._get_all_z_measurements(block_a)
        z_meas_b = self._get_all_z_measurements(block_b)
        
        # For X stabilizers on A, add B's Z measurements as pending transforms
        for x_idx in range(n_x_stabs_a):
            key = (block_a, x_idx)
            if key not in self.pending_x_transforms:
                self.pending_x_transforms[key] = []
            self.pending_x_transforms[key].extend(z_meas_b)
        
        # For X stabilizers on B, add A's Z measurements as pending transforms
        for x_idx in range(n_x_stabs_b):
            key = (block_b, x_idx)
            if key not in self.pending_x_transforms:
                self.pending_x_transforms[key] = []
            self.pending_x_transforms[key].extend(z_meas_a)
    
    def record_cnot_transform(
        self,
        control_block: str,
        target_block: str,
        n_x_stabs_ctrl: int,
        n_z_stabs_tgt: int,
    ) -> None:
        """
        Record stabilizer transforms after a logical CNOT gate.
        
        After CNOT(control, target):
        - X stabilizers on control transform: X_ctrl → X_ctrl ⊗ X_tgt
        - Z stabilizers on target transform: Z_tgt → Z_ctrl ⊗ Z_tgt
        
        Parameters
        ----------
        control_block : str
            Name of control block.
        target_block : str
            Name of target block.
        n_x_stabs_ctrl : int
            Number of X stabilizers in control block.
        n_z_stabs_tgt : int
            Number of Z stabilizers in target block.
        """
        # Get last X measurements from target (for control's X detectors)
        x_meas_tgt = self._get_all_x_measurements(target_block)
        
        # Get last Z measurements from control (for target's Z detectors)
        z_meas_ctrl = self._get_all_z_measurements(control_block)
        
        # For X stabilizers on control, add target's X measurements
        for x_idx in range(n_x_stabs_ctrl):
            key = (control_block, x_idx)
            if key not in self.pending_x_transforms:
                self.pending_x_transforms[key] = []
            self.pending_x_transforms[key].extend(x_meas_tgt)
        
        # For Z stabilizers on target, add control's Z measurements
        for z_idx in range(n_z_stabs_tgt):
            key = (target_block, z_idx)
            if key not in self.pending_z_transforms:
                self.pending_z_transforms[key] = []
            self.pending_z_transforms[key].extend(z_meas_ctrl)
    
    def _get_all_z_measurements(self, block_name: str) -> List[int]:
        """Get all last Z stabilizer measurements for a block."""
        meas = []
        for (bname, stype, sidx), meas_idx in self.last_stabilizer_meas.items():
            if bname == block_name and stype == "z":
                meas.append(meas_idx)
        return meas
    
    def _get_all_x_measurements(self, block_name: str) -> List[int]:
        """Get all last X stabilizer measurements for a block."""
        meas = []
        for (bname, stype, sidx), meas_idx in self.last_stabilizer_meas.items():
            if bname == block_name and stype == "x":
                meas.append(meas_idx)
        return meas
    
    def get_x_detector_measurements(
        self,
        block_name: str,
        stab_idx: int,
        current_meas: int,
    ) -> List[int]:
        """
        Get all measurement indices for an X stabilizer detector.
        
        This includes:
        1. Current measurement
        2. Previous measurement (if exists)
        3. Any pending transform partner measurements (Z from CZ partners)
        
        Parameters
        ----------
        block_name : str
            Block name.
        stab_idx : int
            X stabilizer index.
        current_meas : int
            Current measurement index.
            
        Returns
        -------
        List[int]
            All measurement indices to include in detector.
        """
        meas_list = [current_meas]
        
        # Add previous measurement if exists
        key = (block_name, "x", stab_idx)
        prev = self.last_stabilizer_meas.get(key)
        if prev is not None:
            meas_list.append(prev)
        
        # Add pending transform partner measurements
        transform_key = (block_name, stab_idx)
        transforms = self.pending_x_transforms.get(transform_key, [])
        meas_list.extend(transforms)
        
        return meas_list
    
    def get_z_detector_measurements(
        self,
        block_name: str,
        stab_idx: int,
        current_meas: int,
    ) -> List[int]:
        """
        Get all measurement indices for a Z stabilizer detector.
        
        Similar to X but for Z stabilizers (affected by CNOT transforms).
        
        Parameters
        ----------
        block_name : str
            Block name.
        stab_idx : int
            Z stabilizer index.
        current_meas : int
            Current measurement index.
            
        Returns
        -------
        List[int]
            All measurement indices to include in detector.
        """
        meas_list = [current_meas]
        
        # Add previous measurement if exists
        key = (block_name, "z", stab_idx)
        prev = self.last_stabilizer_meas.get(key)
        if prev is not None:
            meas_list.append(prev)
        
        # Add pending transform partner measurements
        transform_key = (block_name, stab_idx)
        transforms = self.pending_z_transforms.get(transform_key, [])
        meas_list.extend(transforms)
        
        return meas_list
    
    def consume_x_transforms(self, block_name: str, stab_idx: int) -> None:
        """Clear pending X transforms after detector emission."""
        key = (block_name, stab_idx)
        if key in self.pending_x_transforms:
            del self.pending_x_transforms[key]
    
    def consume_z_transforms(self, block_name: str, stab_idx: int) -> None:
        """Clear pending Z transforms after detector emission."""
        key = (block_name, stab_idx)
        if key in self.pending_z_transforms:
            del self.pending_z_transforms[key]
    
    def has_pending_transforms(self, block_name: str, stab_type: str, stab_idx: int) -> bool:
        """Check if a stabilizer has pending transforms."""
        key = (block_name, stab_idx)
        if stab_type == "x":
            return key in self.pending_x_transforms and len(self.pending_x_transforms[key]) > 0
        elif stab_type == "z":
            return key in self.pending_z_transforms and len(self.pending_z_transforms[key]) > 0
        return False
    
    def clear_all_transforms(self, block_name: Optional[str] = None) -> None:
        """Clear all pending transforms, optionally for a specific block."""
        if block_name is None:
            self.pending_x_transforms.clear()
            self.pending_z_transforms.clear()
        else:
            keys_to_remove = [k for k in self.pending_x_transforms if k[0] == block_name]
            for k in keys_to_remove:
                del self.pending_x_transforms[k]
            keys_to_remove = [k for k in self.pending_z_transforms if k[0] == block_name]
            for k in keys_to_remove:
                del self.pending_z_transforms[k]
