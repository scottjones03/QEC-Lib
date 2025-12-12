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
    StabilizerBasis,
    get_logical_support,
)
from qectostim.noise.models import NoiseModel
from qectostim.utils.scheduling_core import graph_coloring_cnots


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


def apply_general_stabilizer_gates_with_ticks(
    circuit: stim.Circuit,
    stab_matrix: np.ndarray,
    data_qubits: List[int],
    ancilla_qubits: List[int],
) -> None:
    """
    Apply stabilizer measurement gates for general (non-CSS) stabilizers with TICK scheduling.
    
    Handles stabilizers with X, Y, and Z components. Uses graph coloring to schedule
    CNOTs into conflict-free layers, with pre/post basis rotation gates.
    
    For each stabilizer component:
    - X: H-CNOT-H
    - Y: S_DAG-H-CNOT-H-S  
    - Z: CNOT
    
    Parameters
    ----------
    circuit : stim.Circuit
        The Stim circuit to append gates to.
    stab_matrix : np.ndarray
        Stabilizer matrix (n_stabs x 2*n_data) in symplectic form [X|Z].
    data_qubits : List[int]
        Global indices of data qubits.
    ancilla_qubits : List[int]
        Global indices of ancilla qubits.
    """
    if stab_matrix is None or stab_matrix.size == 0:
        return
    
    n_stabs = stab_matrix.shape[0]
    n = len(data_qubits)
    
    # Collect all operations: (data_qubit, ancilla, pauli_type)
    # where pauli_type is 'X', 'Y', or 'Z'
    all_ops: List[Tuple[int, int, str]] = []
    
    for s_idx in range(min(n_stabs, len(ancilla_qubits))):
        anc = ancilla_qubits[s_idx]
        x_part = stab_matrix[s_idx, :n]
        z_part = stab_matrix[s_idx, n:2*n] if stab_matrix.shape[1] >= 2*n else np.zeros(n)
        
        for q in range(min(n, len(data_qubits))):
            x_bit = x_part[q] if q < len(x_part) else 0
            z_bit = z_part[q] if q < len(z_part) else 0
            dq = data_qubits[q]
            
            if x_bit and z_bit:
                all_ops.append((dq, anc, 'Y'))
            elif x_bit:
                all_ops.append((dq, anc, 'X'))
            elif z_bit:
                all_ops.append((dq, anc, 'Z'))
    
    if not all_ops:
        return
    
    # Extract just CNOT pairs for scheduling
    cnot_pairs = [(dq, anc) for dq, anc, _ in all_ops]
    
    # Build lookup: (dq, anc) -> pauli_type
    op_pauli = {(dq, anc): p for dq, anc, p in all_ops}
    
    # Schedule into conflict-free layers using shared utility
    layers = graph_coloring_cnots(cnot_pairs)
    
    # Apply each layer with TICK separation
    for layer_idx, layer_cnots in enumerate(layers):
        if layer_idx > 0:
            circuit.append("TICK")
        
        # Group by pauli type for efficient basis changes
        x_ops = [(dq, anc) for dq, anc in layer_cnots if op_pauli.get((dq, anc)) == 'X']
        y_ops = [(dq, anc) for dq, anc in layer_cnots if op_pauli.get((dq, anc)) == 'Y']
        z_ops = [(dq, anc) for dq, anc in layer_cnots if op_pauli.get((dq, anc)) == 'Z']
        
        # Pre-rotation for X components: H
        if x_ops:
            x_data = list(set(dq for dq, _ in x_ops))
            circuit.append("H", x_data)
        
        # Pre-rotation for Y components: S_DAG then H
        if y_ops:
            y_data = list(set(dq for dq, _ in y_ops))
            circuit.append("S_DAG", y_data)
            circuit.append("H", y_data)
        
        # Apply all CNOTs
        for dq, anc in layer_cnots:
            circuit.append("CX", [dq, anc])
        
        # Post-rotation for X components: H
        if x_ops:
            circuit.append("H", x_data)
        
        # Post-rotation for Y components: H then S
        if y_ops:
            circuit.append("H", y_data)
            circuit.append("S", y_data)


def apply_stabilizer_cnots_with_ticks(
    circuit: stim.Circuit,
    parity_check: np.ndarray,
    data_qubits: List[int],
    ancilla_qubits: List[int],
    is_x_type: bool = False,
) -> None:
    """
    Apply stabilizer CNOT gates for CSS codes with TICK scheduling.
    
    This function applies CNOTs for either X-type or Z-type stabilizers,
    scheduling them into conflict-free layers using graph coloring.
    
    For X-type stabilizers, the CNOT direction is data->ancilla.
    For Z-type stabilizers, the CNOT direction is ancilla->data (but we use
    the convention that control is data for Z checks in CSS codes).
    
    Parameters
    ----------
    circuit : stim.Circuit
        The Stim circuit to append gates to.
    parity_check : np.ndarray
        Parity check matrix (n_stabs x n_data) - either Hx or Hz.
    data_qubits : List[int]
        Global indices of data qubits.
    ancilla_qubits : List[int]
        Global indices of ancilla qubits.
    is_x_type : bool
        True for X-type stabilizers (H applied before/after), False for Z-type.
    """
    if parity_check is None or parity_check.size == 0:
        return
    
    n_stabs = parity_check.shape[0]
    n = len(data_qubits)
    
    # Collect all CNOT pairs
    cnot_pairs: List[Tuple[int, int]] = []
    
    for s_idx in range(min(n_stabs, len(ancilla_qubits))):
        anc = ancilla_qubits[s_idx]
        row = parity_check[s_idx]
        
        for q in range(min(n, len(row))):
            if row[q]:
                dq = data_qubits[q]
                if is_x_type:
                    # X-type: CNOT from data to ancilla
                    cnot_pairs.append((dq, anc))
                else:
                    # Z-type: CNOT from ancilla to data
                    cnot_pairs.append((anc, dq))
    
    if not cnot_pairs:
        return
    
    # Schedule into conflict-free layers
    layers = graph_coloring_cnots(cnot_pairs)
    
    # Apply each layer with TICK separation
    for layer_idx, layer_cnots in enumerate(layers):
        if layer_idx > 0:
            circuit.append("TICK")
        
        for ctrl, targ in layer_cnots:
            circuit.append("CX", [ctrl, targ])


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
        if not isinstance(code, StabilizerCode):
            raise TypeError(f"StabilizerMemoryExperiment requires a StabilizerCode, got {type(code)}")
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
    """

    def __init__(
        self,
        code: CSSCode,
        rounds: int,
        noise_model: Dict[str, Any] | None = None,
        basis: str = "Z",
        metadata: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            code=code,
            rounds=rounds,
            noise_model=noise_model,
            basis=basis,
            metadata=metadata
        )

    def to_stim(self) -> stim.Circuit:
        """
        Build a CSS memory experiment using CSSStabilizerRoundBuilder.
        
        Pattern:
          1. Reset data + ancillas
          2. Prepare logical state in chosen basis
          3. Emit stabilizer rounds with time-like detectors
          4. Final data measurement with space-like detectors
          5. Observable declaration
        """
        basis = self.basis.upper()
        
        # Create detector context for tracking
        ctx = DetectorContext()
        
        # Create CSS stabilizer round builder with measurement basis
        builder = CSSStabilizerRoundBuilder(
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
        
        # Emit stabilizer rounds with time-like detectors
        for _ in range(self.rounds):
            builder.emit_round(c, stab_type=StabilizerBasis.BOTH, emit_detectors=True)
        
        # Final measurement and space-like detectors
        builder.emit_final_measurement(c, basis=basis, logical_idx=self.logical_qubit)
        
        # Emit observable
        ctx.emit_observable(c, observable_idx=0)
        
        return c
    
    def to_stim_legacy(self) -> stim.Circuit:
        """
        Build a CSS memory experiment circuit (legacy implementation).
        
        If the code's metadata provides:
          - metadata["x_stab_coords"]    : list of (x, y) for X-check ancillas
          - metadata["z_stab_coords"]    : list of (x, y) for Z-check ancillas
          - metadata["x_schedule"]       : list of (dx, dy) steps
          - metadata["z_schedule"]       : list of (dx, dy) steps

        then we respect that schedule (e.g. rotated surface code with 4 CNOT
        “phases” per stabilizer, moving clockwise).

        Otherwise, we fall back to a naive (but consistent) CSS stabilizer
        circuit:
          - X checks: H on ancilla, CNOT(data->ancilla), H, then measure ancilla.
          - Z checks: CNOT(data->ancilla), then measure ancilla.

        Pattern:
          1. Reset data + ancillas.
          2. Repeat `rounds` times:
             - run X and Z checks,
             - measure ancillas,
             - time-like DETECTORs comparing each ancilla's current vs previous meas.
          3. Measure data once at the end.
          4. Space-like DETECTORs combining last Z-syndrome with data parity.
          5. Single OBSERVABLE_INCLUDE for the logical observable.
        """
        code = self.code
        n = code.n
        hx = code.hx
        hz = code.hz

        # Cache basis as upper-case once.
        basis = self.basis.upper()

        # --- Align layer matrices (hx/hz) with provided geometric coords.
        # Some chain-complex constructions may order/label faces differently,
        # so ensure that the X layer count matches x_stab_coords and likewise for Z.
        meta = getattr(code, "metadata", {}) if hasattr(code, "metadata") else {}
        x_coords_meta = meta.get("x_stab_coords")
        z_coords_meta = meta.get("z_stab_coords")
        if x_coords_meta is not None and z_coords_meta is not None:
            if hx.shape[0] != len(x_coords_meta) and hz.shape[0] == len(x_coords_meta):
                # Swap layers so hx corresponds to X faces and hz to Z faces.
                hx, hz = hz, hx

        n_x = hx.shape[0]
        n_z = hz.shape[0]

        data_qubits = list(range(n))
        anc_x = list(range(n, n + n_x))
        anc_z = list(range(n + n_x, n + n_x + n_z))

        c = stim.Circuit()

        # ---- Geometry / metadata ------------------------------------------
        meta = getattr(code, "metadata", {}) if hasattr(code, "metadata") else {}
        data_coords = meta.get("data_coords")
        x_stab_coords = meta.get("x_stab_coords")
        z_stab_coords = meta.get("z_stab_coords")
        x_schedule = meta.get("x_schedule")
        z_schedule = meta.get("z_schedule")

        coord_to_data: dict[tuple[float, float], int] = {}

        if data_coords is not None:
            for q, (x, y) in enumerate(data_coords):
                coord = (float(x), float(y))
                if coord in coord_to_data:
                    continue
                coord_to_data[coord] = q
                c.append("QUBIT_COORDS", [q], [coord[0], coord[1]])

        if x_stab_coords is not None:
            for a, (x, y) in zip(anc_x, x_stab_coords):
                c.append("QUBIT_COORDS", [a], [float(x), float(y)])

        if z_stab_coords is not None:
            for a, (x, y) in zip(anc_z, z_stab_coords):
                c.append("QUBIT_COORDS", [a], [float(x), float(y)])

        # ---- Initial preparation ------------------------------------------
        total_qubits = n + n_x + n_z
        if total_qubits:
            c.append("R", range(total_qubits))

        # Prepare logical in chosen basis (crude: apply H to all data for X-basis).
        if self.basis.upper() == "X" and n > 0:
            c.append("H", data_qubits)

        c.append("TICK")

        # Measurement index bookkeeping.
        m_index = 0  # total number of measurement results so far
        last_x_meas: list[Optional[int]] = [None] * n_x
        last_z_meas: list[Optional[int]] = [None] * n_z

        def add_detector(coord: tuple[float, float],
                         rec_indices: list[int],
                         t: float = 0.0) -> None:
            """Emit a DETECTOR at space-time coord with rec lookbacks.

            `rec_indices` are absolute measurement indices (0,1,2,...).
            At the moment we call this, `m_index` is the total #meas so far.
            Stim wants lookbacks (negative indices), so we convert via:
                lookback = idx - m_index   (which is <= -1)
            """
            if not rec_indices:
                return
            lookbacks = [idx - m_index for idx in rec_indices]
            c.append(
                "DETECTOR",
                [stim.target_rec(lb) for lb in lookbacks],
                [float(coord[0]), float(coord[1]), t],
            )

        def stab_coord(coords: Optional[List[tuple[float, float]]], idx: int) -> tuple[float, float]:
            if coords is None or idx >= len(coords):
                return (0.0, 0.0)
            x, y = coords[idx]
            return (float(x), float(y))

        use_geo_x = bool(
            x_schedule
            and data_coords is not None
            and x_stab_coords is not None
            and len(x_stab_coords) == n_x
        )
        use_geo_z = bool(
            z_schedule
            and data_coords is not None
            and z_stab_coords is not None
            and len(z_stab_coords) == n_z
        )

        # ---- Syndrome rounds ----------------------------------------------
        interleaved_geo = bool(
            (x_schedule and z_schedule)
            and (data_coords is not None)
            and (x_stab_coords is not None)
            and (z_stab_coords is not None)
            and (len(x_stab_coords) == n_x)
            and (len(z_stab_coords) == n_z)
        )

        for r in range(self.rounds):
            # Prepare ancillas for this round.
            # X ancillas: rotate into |+> at the start of each round.
            if n_x:
                if interleaved_geo or use_geo_x:
                    for s_idx, (sx, sy) in enumerate(x_stab_coords or []):
                        a = anc_x[s_idx]
                        c.append("H", [a])
                else:
                    for s_idx in range(n_x):
                        a = anc_x[s_idx]
                        c.append("H", [a])

            # Z ancillas start in |0> from the initial global reset or the
            # previous round's demolition measurement, so no per-round reset
            # is needed here.

            if interleaved_geo:
                # Interleave X and Z checks per phase (Stim style)
                for (dx_x, dy_x), (dx_z, dy_z) in zip(x_schedule, z_schedule):
                    c.append("TICK")
                    # X layer: CNOT(data -> ancilla)
                    for s_idx, (sx, sy) in enumerate(x_stab_coords):
                        a = anc_x[s_idx]
                        nbr = (float(sx) + dx_x, float(sy) + dy_x)
                        dq = coord_to_data.get(nbr)
                        if dq is not None:
                            c.append("CNOT", [dq, a])
                    # Z layer: CNOT(data -> ancilla)
                    for s_idx, (sx, sy) in enumerate(z_stab_coords):
                        a = anc_z[s_idx]
                        nbr = (float(sx) + dx_z, float(sy) + dy_z)
                        dq = coord_to_data.get(nbr)
                        if dq is not None:
                            c.append("CNOT", [dq, a])
                # Rotate X ancillas back before measurement
                for s_idx in range(n_x):
                    a = anc_x[s_idx]
                    c.append("H", [a])
            else:
                # Fallback: perform X layer then Z layer (non-interleaved)
                # Uses graph-coloring scheduling with TICK separation
                if n_x:
                    if use_geo_x:
                        for dx, dy in x_schedule or []:
                            c.append("TICK")
                            for s_idx, (sx, sy) in enumerate(x_stab_coords or []):
                                a = anc_x[s_idx]
                                nbr = (float(sx) + dx, float(sy) + dy)
                                dq = coord_to_data.get(nbr)
                                if dq is not None:
                                    c.append("CNOT", [dq, a])
                        for s_idx in range(n_x):
                            a = anc_x[s_idx]
                            c.append("H", [a])
                    else:
                        # Use graph-coloring scheduling for proper timing
                        apply_stabilizer_cnots_with_ticks(
                            c, hx, list(range(n)), anc_x, is_x_type=True
                        )
                        # Rotate X ancillas back before measurement
                        for s_idx in range(n_x):
                            a = anc_x[s_idx]
                            c.append("H", [a])
                
                c.append("TICK")  # Separate X and Z stabilizer layers
                
                if n_z:
                    if use_geo_z:
                        for dx, dy in z_schedule or []:
                            c.append("TICK")
                            for s_idx, (sx, sy) in enumerate(z_stab_coords or []):
                                a = anc_z[s_idx]
                                nbr = (float(sx) + dx, float(sy) + dy)
                                dq = coord_to_data.get(nbr)
                                if dq is not None:
                                    c.append("CNOT", [dq, a])
                    else:
                        # Use graph-coloring scheduling for proper timing
                        apply_stabilizer_cnots_with_ticks(
                            c, hz, list(range(n)), anc_z, is_x_type=False
                        )

            # ---- Measure ancillas in batches (like Stim) ----
            # For a CSS memory experiment, we need BOTH X and Z syndromes:
            # - Z-basis memory: Z stabilizers detect X errors (needed for correction)
            #                   X stabilizers also detect Z errors that would flip the observable
            # - X-basis memory: X stabilizers detect Z errors (needed for correction)
            #                   Z stabilizers also detect X errors
            #
            # Both types of stabilizers should be measured in EVERY round to enable
            # proper error correction. The final round may use M instead of MR.
            
            # Collect all X-ancillas to measure in this round
            # X-ancillas are measured every round (not just for X-basis)
            x_ancillas_to_measure = []
            for s_idx in range(n_x):
                # Always measure X-ancillas in every round for proper syndrome extraction
                x_ancillas_to_measure.append((s_idx, anc_x[s_idx]))
            
            # Collect all Z-ancillas to measure in this round
            # Z-ancillas are measured every round for syndrome extraction and space-like detectors
            z_ancillas_to_measure = []
            for s_idx in range(n_z):
                z_ancillas_to_measure.append((s_idx, anc_z[s_idx]))
            
            # Measure X-ancillas in one batch (if any)
            x_meas_start_idx = m_index
            if x_ancillas_to_measure:
                x_qubits = [a for _, a in x_ancillas_to_measure]
                if r < self.rounds - 1:
                    c.append("MR", x_qubits)
                else:
                    c.append("MR", x_qubits)  # Use MR even in last round for consistency
                m_index += len(x_qubits)
            
            # Create time-like detectors for X-ancillas
            # For Z-basis memory, X-ancillas have random first-round outcomes (|0> is not X eigenstate)
            # For X-basis memory, X-ancillas have deterministic first-round outcomes (|+> is X eigenstate)
            for offset, (s_idx, _) in enumerate(x_ancillas_to_measure):
                cur = x_meas_start_idx + offset
                coord = stab_coord(x_stab_coords, s_idx)
                
                if last_x_meas[s_idx] is None:
                    # First round: only create detector if basis matches (X-basis memory)
                    if basis == "X":
                        add_detector(coord, [cur], t=0.0)
                    # else: skip first-round detector for X stabilizers in Z-basis memory
                else:
                    add_detector(coord, [last_x_meas[s_idx], cur], t=0.0)
                
                last_x_meas[s_idx] = cur
            
            # Measure Z-ancillas in one batch (if any)
            z_meas_start_idx = m_index
            if z_ancillas_to_measure:
                z_qubits = [a for _, a in z_ancillas_to_measure]
                c.append("MR", z_qubits)
                m_index += len(z_qubits)
            
            # Create time-like detectors for Z-ancillas
            # For Z-basis memory, Z-ancillas have deterministic first-round outcomes (|0> is Z eigenstate)
            # For X-basis memory, Z-ancillas have random first-round outcomes (|+> is not Z eigenstate)
            for offset, (s_idx, _) in enumerate(z_ancillas_to_measure):
                cur = z_meas_start_idx + offset
                coord = stab_coord(z_stab_coords, s_idx)
                
                if last_z_meas[s_idx] is None:
                    # First round: only create detector if basis matches (Z-basis memory)
                    if basis == "Z":
                        add_detector(coord, [cur], t=0.0)
                    # else: skip first-round detector for Z stabilizers in X-basis memory
                else:
                    add_detector(coord, [last_z_meas[s_idx], cur], t=0.0)
                
                last_z_meas[s_idx] = cur
            
            # Update last_x_meas for all X-ancillas (they are now measured in every round)
            for offset, (s_idx, _) in enumerate(x_ancillas_to_measure):
                last_x_meas[s_idx] = x_meas_start_idx + offset

            if data_coords is not None:
                c.append("SHIFT_COORDS", [], [0.0, 0.0, 1.0])

        # ---- Final data measurement + space-like detectors ----------------
        if n:
            # ALWAYS measure ALL data qubits to ensure space-like detectors have complete information.
            # The logical operator support is used for the observable, not for measurement selection.
            qubits_to_measure = data_qubits
            
            # Measure in chosen basis. For X-basis, rotate with H first.
            if basis == "X":
                c.append("H", qubits_to_measure)
            # Remove disentangling CX gates: do NOT entangle data with ancillas before measurement.
            # Measure data qubits deterministically.
            #
            # REPLACE demolition measurement (MR) with standard M for data qubits.
            c.append("M", qubits_to_measure)

            first_data_idx = m_index
            # Track which qubits were actually measured
            measured_qubits = list(qubits_to_measure)
            data_meas_indices_all = {}  # qubit_index -> measurement_record_index
            for i, q in enumerate(measured_qubits):
                data_meas_indices_all[q] = first_data_idx + i
            m_index += len(measured_qubits)

            # Choose which stabiliser layer to use for *space-like* detectors.
            # We pair the final data measurements with the last round of the
            # same-type stabiliser checks:
            #   - Z-basis memory (|0_L>): use Z checks (hz) and their ancillas.
            #   - X-basis memory (|+_L>): use X checks (hx) and their ancillas.
            if basis == "Z":
                stab_mat = hz
                last_stab_meas = last_z_meas
                stab_coords = z_stab_coords
            else:  # basis == "X"
                stab_mat = hx
                last_stab_meas = last_x_meas
                stab_coords = x_stab_coords

            # Create space-like detectors combining final data measurements with last stabilizer measurements.
            # These detectors represent the parity checks: each stabilizer measures a product of data qubits.
            # We do this regardless of whether the observable covers all stabilizer qubits, since Stim's
            # native implementation also generates these detectors.
            obs_subset = set(measured_qubits) if measured_qubits else set(range(n))
            stab_all_qubits = set()
            if stab_mat is not None and stab_mat.size > 0:
                for row in stab_mat:
                    stab_all_qubits.update(np.where(row == 1)[0])
            
            should_create_space_like = stab_mat is not None and stab_mat.size > 0
            
            if should_create_space_like and stab_mat is not None and stab_mat.size > 0:
                num_stab = stab_mat.shape[0]
                for s_idx in range(num_stab):
                    row = stab_mat[s_idx]
                    # Data measurement indices that participate in this stabiliser.
                    data_idxs = [
                        data_meas_indices_all[q]
                        for q in np.where(row == 1)[0]
                        if q in data_meas_indices_all
                    ]
                    recs = list(data_idxs)
                    # For Z-basis: pair with last Z-ancilla measurements.
                    # For X-basis: pair with last X-ancilla measurements.
                    if basis == "Z":
                        # Z-basis: use Z-ancilla meas (last_z_meas)
                        if s_idx < len(last_z_meas) and last_z_meas[s_idx] is not None:
                            recs.append(last_z_meas[s_idx])
                    else:
                        # X-basis: use X-ancilla meas (last_x_meas)
                        if s_idx < len(last_x_meas) and last_x_meas[s_idx] is not None:
                            recs.append(last_x_meas[s_idx])
                    if not recs:
                        continue
                    # Use correct spacetime coordinates for detector
                    # Note: Stim uses t=1.0 for space-like detectors (final measurement layer)
                    coord = stab_coord(stab_coords, s_idx)
                    add_detector(coord, recs, t=1.0)

            # ---- Logical observable ---------------------------------------
            rec_indices_by_data = data_meas_indices_all

            # Use module-level helper functions (get_logical_ops, pauli_at, ops_valid, ops_len)
            logical_support: list[int] = []
            z_ops = get_logical_ops(code, 'z')
            x_ops = get_logical_ops(code, 'x')
            
            if basis == "Z" and ops_valid(z_ops) and self.logical_qubit < ops_len(z_ops):
                L = z_ops[self.logical_qubit]
                logical_support = [q for q in range(n) if pauli_at(L, q) in ("Z", "Y")]
            elif basis == "X" and ops_valid(x_ops) and self.logical_qubit < ops_len(x_ops):
                L = x_ops[self.logical_qubit]
                logical_support = [q for q in range(n) if pauli_at(L, q) in ("X", "Y")]

            if not logical_support:
                logical_support = measured_qubits

            obs_rec_indices = [
                rec_indices_by_data[q] for q in logical_support if q in rec_indices_by_data
            ]
            if not obs_rec_indices:
                obs_rec_indices = list(data_meas_indices_all.values())

            # Convert absolute indices -> lookbacks for OBSERVABLE_INCLUDE.
            lookbacks = [idx - m_index for idx in obs_rec_indices]
            obs_targets = [stim.target_rec(lb) for lb in lookbacks]
            c.append("OBSERVABLE_INCLUDE", obs_targets, 0)

        # NOTE: Final measurement block on data qubits uses M (not MR).
        #       Detector coordinates above use correct spacetime (x, y, t).
        #       All DETECTOR rec indices reference valid measurement results.

        return c


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
