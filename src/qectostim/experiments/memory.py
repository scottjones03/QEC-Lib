from __future__ import annotations

import abc
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import stim

from qectostim.codes.abstract_code import Code, StabilizerCode, PauliString
from qectostim.codes.abstract_css import CSSCode
from qectostim.experiments.experiment import Experiment
from qectostim.experiments.stabilizer_rounds import (
    DetectorContext,
    StabilizerRoundBuilder,
    CSSStabilizerRoundBuilder,
    ColorCodeStabilizerRoundBuilder,
    GeneralStabilizerRoundBuilder,
    XYZColorCodeStabilizerRoundBuilder,
    StabilizerBasis,
    get_logical_support,
)
from qectostim.noise.models import NoiseModel
# ============================================================================
# Helper Functions (shared across experiment classes)
# ============================================================================

def get_logical_ops(code, op_type: str):
    """Get logical operators, handling both property and method access."""
    ops_attr = getattr(code, f'logical_{op_type}_ops', None)
    if ops_attr is None:
        return []
    return ops_attr() if callable(ops_attr) else ops_attr


def ops_valid(ops) -> bool:
    """Check if logical ops exist and are non-empty."""
    if ops is None:
        return False
    if isinstance(ops, np.ndarray):
        return ops.size > 0
    return bool(len(ops) > 0)


def ops_len(ops) -> int:
    """Get length of logical ops."""
    if isinstance(ops, np.ndarray):
        return ops.shape[0] if ops.ndim > 0 else 0
    return len(ops)


def pauli_at(pauli_obj, q: int) -> str:
    """Get Pauli at qubit q. Returns 'I' if not in support."""
    if isinstance(pauli_obj, str):
        if q < len(pauli_obj):
            return pauli_obj[q]
        return 'I'
    elif isinstance(pauli_obj, dict):
        return pauli_obj.get(q, 'I')
    return 'I'





# ============================================================================
# Base Classes with Template Method Helpers
# ============================================================================

class MemoryExperiment(Experiment):
    """
    Repeated stabilizer measurement (memory) experiment for a single logical qubit.
    
    This base class provides shared helper methods for circuit construction
    that subclasses can use via the template method pattern.
    """

    def __init__(
        self,
        code: Code,
        noise_model: NoiseModel | None,
        rounds: int,
        logical_qubit: int = 0,
        initial_state: str = "0",
        metadata: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(code, noise_model, metadata)
        self.rounds = rounds
        self.logical_qubit = logical_qubit
        self.initial_state = initial_state
        self.operation = "memory"

    @abc.abstractmethod
    def to_stim(self) -> stim.Circuit:
        ...
    
    # ========================================================================
    # Template Method Helpers - Shared Circuit Building Logic
    # ========================================================================
    
    def _emit_qubit_coords(
        self,
        circuit: stim.Circuit,
        data_qubits: List[int],
        data_coords: Optional[List[Tuple[float, float]]],
        ancilla_qubits: Optional[List[int]] = None,
        ancilla_coords: Optional[List[Tuple[float, float]]] = None,
    ) -> Dict[Tuple[float, float], int]:
        """Emit QUBIT_COORDS instructions for data and ancilla qubits.
        
        Returns a mapping from (x, y) coordinate to data qubit index.
        """
        coord_to_data: Dict[Tuple[float, float], int] = {}
        
        if data_coords is not None:
            for q, coord in zip(data_qubits, data_coords):
                if len(coord) >= 2:
                    x, y = float(coord[0]), float(coord[1])
                    coord_tuple = (x, y)
                    if coord_tuple not in coord_to_data:
                        coord_to_data[coord_tuple] = q
                        circuit.append("QUBIT_COORDS", [q], [x, y])
        
        if ancilla_qubits is not None and ancilla_coords is not None:
            for a, coord in zip(ancilla_qubits, ancilla_coords):
                if len(coord) >= 2:
                    circuit.append("QUBIT_COORDS", [a], [float(coord[0]), float(coord[1])])
        
        return coord_to_data
    
    def _add_detector(
        self,
        circuit: stim.Circuit,
        rec_indices: List[int],
        m_index: int,
        coord: Tuple[float, ...] = (0.0, 0.0, 0.0),
    ) -> None:
        """Emit a DETECTOR with the given measurement record indices.
        
        Args:
            circuit: The stim circuit to append to.
            rec_indices: Absolute measurement indices (0, 1, 2, ...).
            m_index: Current total number of measurements (for computing lookbacks).
            coord: Detector coordinates (x, y, t) or (x, y, t, color).
        """
        if not rec_indices:
            return
        lookbacks = [idx - m_index for idx in rec_indices]
        circuit.append(
            "DETECTOR",
            [stim.target_rec(lb) for lb in lookbacks],
            list(coord),
        )
    
    def _emit_observable(
        self,
        circuit: stim.Circuit,
        rec_indices: List[int],
        m_index: int,
        observable_index: int = 0,
    ) -> None:
        """Emit an OBSERVABLE_INCLUDE declaration.
        
        Args:
            circuit: The stim circuit to append to.
            rec_indices: Absolute measurement indices for the logical observable.
            m_index: Current total number of measurements.
            observable_index: Which logical observable (default 0).
        """
        if not rec_indices:
            return
        lookbacks = [idx - m_index for idx in rec_indices]
        obs_targets = [stim.target_rec(lb) for lb in lookbacks]
        circuit.append("OBSERVABLE_INCLUDE", obs_targets, observable_index)
    
    def _get_detector_coords(
        self,
        stab_coords: Optional[List[Tuple[float, float]]],
        s_idx: int,
        t: float,
    ) -> Tuple[float, ...]:
        """Return detector coordinates for a stabilizer.
        
        Subclasses can override to add additional coordinates (e.g., color).
        
        Args:
            stab_coords: List of (x, y) stabilizer positions.
            s_idx: Stabilizer index.
            t: Time coordinate.
            
        Returns:
            Tuple of detector coordinates (x, y, t) or extended.
        """
        if stab_coords is None or s_idx >= len(stab_coords):
            return (0.0, 0.0, t)
        x, y = stab_coords[s_idx]
        return (float(x), float(y), t)
    
    def _get_logical_support(
        self,
        code: Code,
        basis: str,
        logical_qubit: int,
        n: int,
    ) -> List[int]:
        """Get the qubit support for a logical operator.
        
        Args:
            code: The quantum error correcting code.
            basis: "Z" or "X" - which logical to use.
            logical_qubit: Index of the logical qubit.
            n: Total number of physical qubits.
            
        Returns:
            List of qubit indices in the logical operator support.
        """
        if basis == "Z":
            ops = get_logical_ops(code, 'z')
            pauli_chars = ("Z", "Y")
        else:
            ops = get_logical_ops(code, 'x')
            pauli_chars = ("X", "Y")
        
        if ops_valid(ops) and logical_qubit < ops_len(ops):
            L = ops[logical_qubit]
            return [q for q in range(n) if pauli_at(L, q) in pauli_chars]
        return []


class StabilizerMemoryExperiment(MemoryExperiment):
    """
    Memory experiment for general stabilizer codes (both CSS and non-CSS).
    
    This base class handles mixed X/Z stabilizers by directly using the
    symplectic stabilizer_matrix representation. Each stabilizer is measured
    by entangling an ancilla with all data qubits in the stabilizer's support.
    
    For CSS codes, consider using CSSMemoryExperiment which takes advantage
    of the separate X/Z structure for more efficient circuits.
    """
    
    def __init__(
        self,
        code: StabilizerCode,
        rounds: int,
        noise_model: Dict[str, Any] | None = None,
        basis: str = "Z",
        metadata: Optional[Dict[str, Any]] = None,
    ):
        # Use duck typing instead of isinstance to handle module reload issues
        # Check for required attributes that all stabilizer codes must have
        required_attrs = ['n', 'k']
        missing = [attr for attr in required_attrs if not hasattr(code, attr)]
        if missing:
            raise TypeError(f"StabilizerMemoryExperiment requires a code with {missing} attributes, got {type(code)}")
        # Parent MemoryExperiment.__init__ has signature: (code, noise_model, rounds, ...)
        super().__init__(code, noise_model, rounds, metadata=metadata)
        self.basis = basis  # "Z" or "X"
    
    def to_stim(self) -> stim.Circuit:
        """
        Build a memory experiment circuit for a general stabilizer code.
        
        Uses GeneralStabilizerRoundBuilder for consistent circuit construction
        with proper CNOT scheduling and detector generation.
        
        The circuit structure:
        1. Reset all data qubits and ancillas
        2. Prepare logical state in chosen basis
        3. Emit stabilizer rounds with time-like detectors
        4. Final data measurement with space-like detectors
        5. Observable declaration
        """
        basis = self.basis.upper()
        
        # Create detector context for tracking
        ctx = DetectorContext()
        
        # Create general stabilizer round builder
        builder = GeneralStabilizerRoundBuilder(
            self.code, ctx, 
            block_name="main",
        )
        
        c = stim.Circuit()
        
        # Emit qubit coordinates
        builder.emit_qubit_coords(c)
        
        # Reset all qubits
        builder.emit_reset_all(c)
        
        # Prepare logical state
        initial_state = "+" if basis == "X" else "0"
        builder.emit_prepare_logical_state(c, state=initial_state, logical_idx=self.logical_qubit)
        
        # Emit stabilizer rounds with time-like detectors
        for _ in range(self.rounds):
            builder.emit_round(c, emit_detectors=True)
        
        # Final measurement and space-like detectors
        builder.emit_final_measurement(c, basis=basis, logical_idx=self.logical_qubit)
        
        # Emit observable
        ctx.emit_observable(c, observable_idx=0)
        
        return c


class CSSMemoryExperiment(StabilizerMemoryExperiment):
    """
    Optimized memory experiment for CSS codes.
    
    Uses StabilizerRoundBuilder for efficient circuit construction with
    proper scheduling and detector generation.
    
    Parameters
    ----------
    code : CSSCode
        The CSS code to use for the memory experiment.
    rounds : int
        Number of syndrome measurement rounds.
    noise_model : Dict[str, Any] | None
        Noise model parameters.
    basis : str
        Measurement basis ("Z" or "X").
    enable_metachecks : bool
        If True and the code has metachecks (e.g., 4D codes), emit metacheck
        detectors that enable single-shot error correction.
    single_shot_metachecks : bool
        If True (and enable_metachecks is True), use spatial metacheck mode
        where detectors fire on syndrome constraint violations within a single
        round. This is true single-shot QEC. If False, uses time-like mode
        comparing consecutive rounds (less effective).
    metadata : Dict[str, Any] | None
        Additional metadata.
    """

    def __init__(
        self,
        code: CSSCode,
        rounds: int,
        noise_model: Dict[str, Any] | None = None,
        basis: str = "Z",
        enable_metachecks: bool = False,
        single_shot_metachecks: bool = True,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            code=code,
            rounds=rounds,
            noise_model=noise_model,
            basis=basis,
            metadata=metadata
        )
        self.enable_metachecks = enable_metachecks
        self.single_shot_metachecks = single_shot_metachecks

    def to_stim(self) -> stim.Circuit:
        """
        Build a CSS memory experiment using CSSStabilizerRoundBuilder.
        
        Pattern:
          1. Reset data + ancillas
          2. Prepare logical state in chosen basis
          3. First stabilizer round (separate for initialization detectors)
          4. REPEAT block for subsequent rounds (enables DEM time-translation invariance)
          5. Final data measurement with space-like detectors
          6. Observable declaration
          
        Note: Using REPEAT blocks is essential for matching Stim's DEM error
        mechanism counts. Without REPEAT, each round gets unique detector IDs
        and the DEM can't recognize time-translation-invariant error patterns.
        """
        basis = self.basis.upper()
        
        # Create detector context for tracking
        ctx = DetectorContext()
        
        # Create CSS stabilizer round builder with measurement basis
        # Enable metachecks if requested and code supports them
        use_metachecks = self.enable_metachecks and getattr(self.code, 'has_metachecks', False)
        builder = CSSStabilizerRoundBuilder(
            self.code, ctx, 
            block_name="main",
            measurement_basis=basis,
            enable_metachecks=use_metachecks,
            single_shot_metachecks=self.single_shot_metachecks
        )
        
        c = stim.Circuit()
        
        # Emit qubit coordinates
        builder.emit_qubit_coords(c)
        
        # Reset all qubits
        builder.emit_reset_all(c)
        
        # Prepare logical state
        initial_state = "+" if basis == "X" else "0"
        builder.emit_prepare_logical_state(c, state=initial_state, logical_idx=self.logical_qubit)
        
        # First stabilizer round (has special first-round detector logic)
        builder.emit_round(c, stab_type=StabilizerBasis.BOTH, emit_detectors=True, emit_metachecks=use_metachecks)
        
        # Subsequent rounds in REPEAT block for DEM time-translation invariance
        if self.rounds > 1:
            repeat_body = stim.Circuit()
            builder.emit_round(repeat_body, stab_type=StabilizerBasis.BOTH, emit_detectors=True, emit_metachecks=use_metachecks)
            c.append(stim.CircuitRepeatBlock(self.rounds - 1, repeat_body))
        
        # Final measurement and space-like detectors
        builder.emit_final_measurement(c, basis=basis, logical_idx=self.logical_qubit)
        
        # Emit observable
        ctx.emit_observable(c, observable_idx=0)
        
        return c

    # REMOVED: to_stim_legacy method (legacy code, ~430 lines)
    # The method was removed as part of code cleanup - it was never called.

class ColorCodeMemoryExperiment(CSSMemoryExperiment):
    """
    Memory experiment for color codes with Chromobius-compatible DEM generation.
    
    This class extends CSSMemoryExperiment to emit detector coordinates with the
    4th component encoding (basis, color) as required by the Chromobius decoder:
    
    - coord[3] = 0, 1, 2 for X-type red, green, blue stabilizers
    - coord[3] = 3, 4, 5 for Z-type red, green, blue stabilizers
    
    The code must provide:
    - metadata["stab_colors"]: list of colors (0=red, 1=green, 2=blue) per stabilizer
    - metadata["is_chromobius_compatible"]: True
    
    For self-dual color codes (where X and Z stabilizers have the same support),
    both X and Z detectors use the same color assignments.
    """
    
    def __init__(
        self,
        code: CSSCode,
        rounds: int,
        noise_model: Dict[str, Any] | None = None,
        basis: str = "Z",
        metadata: Optional[Dict[str, Any]] = None,
    ):
        # Validate that code has color metadata
        code_meta = getattr(code, "metadata", {}) if hasattr(code, "metadata") else {}
        if not code_meta.get("is_chromobius_compatible", False):
            raise ValueError(
                "ColorCodeMemoryExperiment requires a code with "
                "metadata['is_chromobius_compatible'] = True"
            )
        if "stab_colors" not in code_meta:
            raise ValueError(
                "ColorCodeMemoryExperiment requires code.metadata['stab_colors'] "
                "to be a list of colors (0=red, 1=green, 2=blue) per stabilizer"
            )
        
        super().__init__(
            code=code,
            rounds=rounds,
            noise_model=noise_model,
            basis=basis,
            metadata=metadata
        )
        
        self._stab_colors = code_meta["stab_colors"]
    
    def to_stim(self) -> stim.Circuit:
        """
        Build a color code memory experiment using ColorCodeStabilizerRoundBuilder.
        
        Uses the specialized builder that emits 4D detector coordinates
        with color encoding for Chromobius compatibility.
        """
        basis = self.basis.upper()
        
        # Create detector context for tracking
        ctx = DetectorContext()
        
        # Create color code stabilizer round builder
        builder = ColorCodeStabilizerRoundBuilder(
            self.code, ctx,
            block_name="main",
            measurement_basis=basis
        )
        
        c = stim.Circuit()
        
        # Emit qubit coordinates
        builder.emit_qubit_coords(c)
        
        # Reset all qubits
        builder.emit_reset_all(c)
        
        # Prepare logical state
        initial_state = "+" if basis == "X" else "0"
        builder.emit_prepare_logical_state(c, state=initial_state, logical_idx=self.logical_qubit)
        
        # Emit stabilizer rounds with time-like detectors (4D coords with color)
        for _ in range(self.rounds):
            builder.emit_round(c, stab_type=StabilizerBasis.BOTH, emit_detectors=True)
        
        # Final measurement and space-like detectors (4D coords with color)
        builder.emit_final_measurement(c, basis=basis, logical_idx=self.logical_qubit)
        
        # Emit observable
        ctx.emit_observable(c, observable_idx=0)
        
        return c


class XYZColorCodeMemoryExperiment(CSSMemoryExperiment):
    """
    Memory experiment for XYZ color codes with C_XYZ basis cycling.
    
    This experiment uses the XYZColorCodeStabilizerRoundBuilder to create
    circuits that apply C_XYZ gates each round, cycling the measurement
    basis through X→Y→Z→X.
    
    Unlike CSS color codes, XYZ codes:
    - Use one ancilla per face (not separate X/Z)
    - Apply C_XYZ to data qubits each round
    - Have joint XYZ measurements rather than separate X and Z
    
    Chromobius Compatible:
    - Emits 4D detector coordinates (x, y, t, color)
    - Color ∈ {0, 1, 2} (single basis encoding)
    
    Parameters
    ----------
    code : Code
        An XYZ color code with appropriate metadata.
    rounds : int
        Number of syndrome extraction rounds.
    noise_model : Dict or None
        Noise parameters.
    basis : str
        Memory basis ("Z" or "X").
    metadata : Dict or None
        Additional experiment metadata.
    
    Example
    -------
    >>> from qectostim.codes.color.triangular_colour_xyz import TriangularColourCodeXYZ
    >>> from qectostim.experiments.memory import XYZColorCodeMemoryExperiment
    >>> 
    >>> code = TriangularColourCodeXYZ(d=3)
    >>> exp = XYZColorCodeMemoryExperiment(code, rounds=3)
    >>> circuit = exp.to_stim()
    >>> # Decode with Chromobius
    """
    
    def __init__(
        self,
        code: Code,
        rounds: int,
        noise_model: Dict[str, Any] | None = None,
        basis: str = "Z",
        metadata: Optional[Dict[str, Any]] = None,
    ):
        # Validate XYZ color code metadata
        code_meta = getattr(code, "metadata", {}) if hasattr(code, "metadata") else {}
        
        if not code_meta.get("is_chromobius_compatible", False):
            raise ValueError(
                "XYZColorCodeMemoryExperiment requires a code with "
                "metadata['is_chromobius_compatible'] = True"
            )
        if "faces" not in code_meta:
            raise ValueError(
                "XYZColorCodeMemoryExperiment requires code.metadata['faces'] "
                "to be a list of data qubit indices per stabilizer face"
            )
        if "stab_colors" not in code_meta:
            raise ValueError(
                "XYZColorCodeMemoryExperiment requires code.metadata['stab_colors'] "
                "to be a list of colors (0=red, 1=green, 2=blue) per face"
            )
        if "ancilla_coords" not in code_meta:
            raise ValueError(
                "XYZColorCodeMemoryExperiment requires code.metadata['ancilla_coords'] "
                "to be a list of (x, y) coordinates for ancilla qubits"
            )
        
        # Call StabilizerMemoryExperiment init to set attributes correctly
        # This sets: code, noise_model, rounds, metadata, basis, logical_qubit, initial_state
        StabilizerMemoryExperiment.__init__(
            self,
            code=code,
            rounds=rounds,
            noise_model=noise_model,
            basis=basis,
            metadata=metadata
        )
        
        self._faces = code_meta["faces"]
        self._stab_colors = code_meta["stab_colors"]
        self._ancilla_coords = code_meta["ancilla_coords"]
    
    def to_stim(self) -> stim.Circuit:
        """
        Build an XYZ color code memory experiment circuit.
        
        Following Stim's color_code:memory_xyz structure exactly.
        The circuit structure varies based on `rounds % 3`:
        
        For rounds % 3 == 0 (e.g., 3, 6, 9):
            1. REPEAT 2 { syndrome round }
            2. 2-way detectors  
            3. One unrolled round with triple XOR detectors
            4. M (Z measurement)
            5. Final detectors with 1 syndrome ref
        
        For rounds % 3 == 1 (e.g., 4, 7):
            1. REPEAT 2 { syndrome round }
            2. 2-way detectors
            3. REPEAT (rounds-2) { round + triple XOR }
            4. MX (X measurement)
            5. Final detectors with 2 syndrome refs
        
        For rounds % 3 == 2 (e.g., 5, 8):
            1. REPEAT 2 { syndrome round }
            2. 2-way detectors
            3. REPEAT (rounds-2) { round + triple XOR }
            4. MY (Y measurement)
            5. Final detectors with 2 syndrome refs
        
        Note: XYZ codes use 3D coordinates (x, y, t), NOT 4D.
        This means they work with PyMatching, NOT Chromobius.
        """
        basis = self.basis.upper()
        
        # Get code info
        n_data = self.code.n
        n_faces = len(self._faces)
        n_total = n_data + n_faces
        coords = self.code.metadata.get("data_coords", self.code.metadata.get("coords", [(0, 0)] * n_data))
        ancilla_coords = self._ancilla_coords
        stab_colors = self._stab_colors
        
        if self.rounds < 2:
            raise ValueError("XYZ color code requires rounds >= 2")
        
        c = stim.Circuit()
        
        # Qubit coordinates (data then ancilla)
        for i in range(n_data):
            x, y = coords[i] if i < len(coords) else (0, 0)
            c.append("QUBIT_COORDS", [i], [float(x), float(y)])
        for i, (x, y) in enumerate(ancilla_coords):
            c.append("QUBIT_COORDS", [n_data + i], [float(x), float(y)])
        
        # Reset all
        c.append("R", list(range(n_total)))
        
        # Build CNOT schedule
        cnot_layers = self._build_cnot_schedule()
        
        # Helper: build one syndrome round
        def build_syndrome_round() -> stim.Circuit:
            rc = stim.Circuit()
            rc.append("TICK")
            rc.append("C_XYZ", list(range(n_data)))
            rc.append("TICK")
            
            for layer in cnot_layers:
                if layer:
                    targets = []
                    for data_q, anc_idx in layer:
                        targets.extend([data_q, n_data + anc_idx])
                    rc.append("CX", targets)
                rc.append("TICK")
            
            rc.append("MR", list(range(n_data, n_total)))
            return rc
        
        phase = self.rounds % 3
        
        # REPEAT 2 for first 2 rounds (always)
        repeat_init = build_syndrome_round()
        c.append(stim.CircuitRepeatBlock(2, repeat_init))
        
        # 2-way detectors after first REPEAT
        for a_idx in range(n_faces):
            x, y = ancilla_coords[a_idx]
            rec_curr = -(n_faces - a_idx)
            rec_prev = rec_curr - n_faces
            c.append("DETECTOR", 
                    [stim.target_rec(rec_curr), stim.target_rec(rec_prev)],
                    [float(x), float(y), 0.0])
        
        if phase == 0:
            # rounds % 3 == 0: One unrolled round + M + 1 syndrome ref in final detector
            # Unrolled round
            unrolled = build_syndrome_round()
            for instr in unrolled:
                c.append(instr)
            
            c.append("SHIFT_COORDS", [], [0, 0, 1])
            
            # Triple XOR detectors
            for a_idx in range(n_faces):
                x, y = ancilla_coords[a_idx]
                rec_r = -(n_faces - a_idx)
                rec_r1 = rec_r - n_faces
                rec_r2 = rec_r1 - n_faces
                c.append("DETECTOR",
                        [stim.target_rec(rec_r), stim.target_rec(rec_r1), stim.target_rec(rec_r2)],
                        [float(x), float(y), 0.0])
            
            # Additional rounds if needed (for rounds=6, 9, etc.)
            extra_rounds = (self.rounds - 3) // 3
            if extra_rounds > 0:
                repeat_body = build_syndrome_round()
                repeat_body.append("SHIFT_COORDS", [], [0, 0, 1])
                for a_idx in range(n_faces):
                    x, y = ancilla_coords[a_idx]
                    rec_r = -(n_faces - a_idx)
                    rec_r1 = rec_r - n_faces
                    rec_r2 = rec_r1 - n_faces
                    repeat_body.append("DETECTOR",
                            [stim.target_rec(rec_r), stim.target_rec(rec_r1), stim.target_rec(rec_r2)],
                            [float(x), float(y), 0.0])
                c.append(stim.CircuitRepeatBlock(extra_rounds * 3, repeat_body))
            
            # M measurement
            c.append("M", list(range(n_data)))
            
            # Final detectors with 1 syndrome ref
            for a_idx, face in enumerate(self._faces):
                x, y = ancilla_coords[a_idx]
                rec_syndrome = -(n_data + n_faces - a_idx)
                
                recs = []
                for dq in face:
                    recs.append(stim.target_rec(-(n_data - dq)))
                recs.append(stim.target_rec(rec_syndrome))
                
                c.append("DETECTOR", recs, [float(x), float(y), 1.0])
        
        else:
            # rounds % 3 == 1 or 2: REPEAT (rounds-2) with triple XOR inside
            if self.rounds > 2:
                repeat_body = build_syndrome_round()
                repeat_body.append("SHIFT_COORDS", [], [0, 0, 1])
                
                for a_idx in range(n_faces):
                    x, y = ancilla_coords[a_idx]
                    rec_r = -(n_faces - a_idx)
                    rec_r1 = rec_r - n_faces
                    rec_r2 = rec_r1 - n_faces
                    repeat_body.append("DETECTOR",
                            [stim.target_rec(rec_r), stim.target_rec(rec_r1), stim.target_rec(rec_r2)],
                            [float(x), float(y), 0.0])
                
                c.append(stim.CircuitRepeatBlock(self.rounds - 2, repeat_body))
            
            # MX or MY measurement
            if phase == 1:
                c.append("MX", list(range(n_data)))
            else:  # phase == 2
                c.append("MY", list(range(n_data)))
            
            # Final detectors
            # For phase == 1 (MX): 1 syndrome ref from second-to-last round
            #   - Last round (rounds-1) syndrome is at rec[-(n_data + n_faces - a_idx)]
            #   - Second-to-last (rounds-2) syndrome is at rec[-(n_data + 2*n_faces - a_idx)]
            # For phase == 2 (MY): 2 syndrome refs from both last and second-to-last rounds
            for a_idx, face in enumerate(self._faces):
                x, y = ancilla_coords[a_idx]
                # Last round syndrome position
                rec_syndrome_last = -(n_data + n_faces - a_idx)
                # Second-to-last round syndrome position
                rec_syndrome_prev = rec_syndrome_last - n_faces
                
                recs = []
                for dq in face:
                    recs.append(stim.target_rec(-(n_data - dq)))
                
                if phase == 1:
                    # MX case: use second-to-last round syndrome only
                    recs.append(stim.target_rec(rec_syndrome_prev))
                else:  # phase == 2
                    # MY case: use both last and second-to-last round syndromes
                    recs.append(stim.target_rec(rec_syndrome_last))
                    recs.append(stim.target_rec(rec_syndrome_prev))
                
                c.append("DETECTOR", recs, [float(x), float(y), 1.0])
        
        # Observable
        if basis == "Z":
            logical_support = self.code.metadata.get("logical_z_support", list(range(min(3, n_data))))
        else:
            logical_support = self.code.metadata.get("logical_x_support", list(range(min(3, n_data))))
        
        obs_recs = [stim.target_rec(-(n_data - q)) for q in logical_support]
        c.append("OBSERVABLE_INCLUDE", obs_recs, [0])
        
        return c
    
    def _build_cnot_schedule(self) -> List[List[Tuple[int, int]]]:
        """Build CNOT schedule in layers to avoid qubit conflicts."""
        all_cnots = []
        for anc_idx, face in enumerate(self._faces):
            for data_q in face:
                all_cnots.append((data_q, anc_idx))
        
        layers = []
        for data_q, anc_idx in all_cnots:
            placed = False
            for layer in layers:
                conflict = any(d == data_q or a == anc_idx for d, a in layer)
                if not conflict:
                    layer.append((data_q, anc_idx))
                    placed = True
                    break
            if not placed:
                layers.append([(data_q, anc_idx)])
        
        return layers
