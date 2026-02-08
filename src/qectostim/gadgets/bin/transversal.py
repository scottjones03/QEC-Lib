# src/qectostim/gadgets/transversal.py
"""
Transversal logical gate gadgets.

Transversal gates are logical gates implemented by applying physical gates
to each qubit independently (single-qubit) or to corresponding qubit pairs
(two-qubit gates between code blocks). This is the simplest form of
fault-tolerant logical gate since errors don't spread between qubits.

Supported gates:
- Single-qubit: H, S, S_DAG, T, T_DAG, X, Y, Z
- Two-qubit: CNOT, CZ (between two code blocks)

Example usage:
    >>> from qectostim.gadgets.transversal import TransversalHadamard, TransversalCNOT
    >>> from qectostim.codes.surface import SurfaceCode
    >>> from qectostim.noise.models import UniformDepolarizing
    >>>
    >>> code = SurfaceCode(distance=3)
    >>> noise = UniformDepolarizing(p=0.001)
    >>> 
    >>> # Single-qubit H gate
    >>> h_gadget = TransversalHadamard()
    >>> circuit = h_gadget.to_stim([code], noise)
    >>>
    >>> # Two-qubit CNOT between two code blocks
    >>> code1, code2 = SurfaceCode(distance=3), SurfaceCode(distance=3)
    >>> cnot_gadget = TransversalCNOT()
    >>> circuit = cnot_gadget.to_stim([code1, code2], noise)
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import (
    List,
    Dict,
    Any,
    Optional,
    Tuple,
    Set,
    Union,
    Callable,
)

import stim

from qectostim.codes.abstract_code import Code
from qectostim.noise.models import NoiseModel
from qectostim.gadgets.base import (
    Gadget,
    GadgetMetadata,
    StabilizerTransform,
    ObservableTransform,
    TwoQubitObservableTransform,
    TransversalGadgetMixin,
    PhaseResult,
    PhaseType,
)
from qectostim.gadgets.coordinates import (
    CoordND,
    get_code_dimension,
    get_bounding_box,
    translate_coords,
    pad_coord_to_dim,
    emit_qubit_coords_nd,
    get_code_coords,
    compute_non_overlapping_offset,
)
from qectostim.gadgets.layout import (
    GadgetLayout,
    BlockInfo,
    QubitIndexMap,
    QubitAllocation,
)

# TYPE_CHECKING imports
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from qectostim.experiments.stabilizer_rounds import DetectorContext


# Gate name to Stim instruction mapping
# Note: T and T_DAG are NOT supported by Stim (non-Clifford gates)
# They are included here for documentation but will fail at runtime
STIM_GATE_MAP = {
    "H": "H",
    "S": "S",
    "S_DAG": "S_DAG",
    # T gates are non-Clifford and cannot be simulated in Stim
    # Use magic state distillation or teleportation-based T in practice
    # "T": "T",
    # "T_DAG": "T_DAG",
    "X": "X",
    "Y": "Y",
    "Z": "Z",
    "CNOT": "CNOT",
    "CX": "CNOT",
    "CZ": "CZ",
    "SWAP": "SWAP",
}

# Single-qubit gates
SINGLE_QUBIT_GATES = {"H", "S", "S_DAG", "T", "T_DAG", "X", "Y", "Z"}

# Two-qubit gates (require two code blocks)
TWO_QUBIT_GATES = {"CNOT", "CX", "CZ", "SWAP"}


class TransversalGate(Gadget, TransversalGadgetMixin):
    """
    Base class for transversal gate gadgets.
    
    A transversal gate applies physical gates to all qubits in parallel,
    either single-qubit gates to one code block, or two-qubit gates
    between corresponding qubits of two code blocks.
    
    Attributes:
        gate_name: The name of the gate (H, S, T, CNOT, CZ, etc.)
        include_stabilizer_rounds: Whether to include stabilizer measurement
            rounds before/after the gate
        num_rounds_before: Number of stabilizer rounds before gate
        num_rounds_after: Number of stabilizer rounds after gate
    """
    
    def __init__(
        self,
        gate_name: str,
        input_state: str = "0",
        include_stabilizer_rounds: bool = True,
        num_rounds_before: int = 1,
        num_rounds_after: int = 1,
    ):
        """
        Initialize transversal gate gadget.
        
        Args:
            gate_name: Name of gate (H, S, T, CNOT, CZ, etc.)
            input_state: Logical input state ("0" or "+")
            include_stabilizer_rounds: Include stabilizer measurements
            num_rounds_before: Stabilizer rounds before gate
            num_rounds_after: Stabilizer rounds after gate
        """
        super().__init__(input_state=input_state)
        
        if gate_name not in STIM_GATE_MAP:
            raise ValueError(
                f"Unknown gate '{gate_name}'. Supported: {list(STIM_GATE_MAP.keys())}"
            )
        
        self.gate_name = gate_name  # Override base class default
        self.stim_gate = STIM_GATE_MAP[gate_name]
        self.is_two_qubit = gate_name in TWO_QUBIT_GATES
        self.include_stabilizer_rounds = include_stabilizer_rounds
        self.num_rounds_before = num_rounds_before
        self.num_rounds_after = num_rounds_after
        
        # Cached results
        self._last_codes: Optional[List[Code]] = None
        self._detector_idx: int = 0
        self._current_time: float = 0.0
    
    def validate_codes(self, codes: List[Code]) -> None:
        """Validate that codes are compatible with this gate."""
        super().validate_codes(codes)
        
        if self.is_two_qubit:
            if len(codes) != 2:
                raise ValueError(
                    f"Two-qubit gate {self.gate_name} requires exactly 2 codes, "
                    f"got {len(codes)}"
                )
            if codes[0].n != codes[1].n:
                raise ValueError(
                    f"Codes must have same number of data qubits for transversal "
                    f"{self.gate_name}. Got {codes[0].n} and {codes[1].n}"
                )
        else:
            if len(codes) != 1:
                raise ValueError(
                    f"Single-qubit gate {self.gate_name} requires exactly 1 code, "
                    f"got {len(codes)}"
                )
        
        # Check if gate is supported
        for code in codes:
            if not self.check_transversal_support(code, self.gate_name):
                # Warning but don't fail - might be intentional testing
                pass
    
    # =========================================================================
    # PRIMARY INTERFACE - Required by FaultTolerantGadgetExperiment
    # =========================================================================
    
    @property
    def num_phases(self) -> int:
        """Transversal gates are single-phase."""
        return 1
    
    def emit_next_phase(
        self,
        circuit: stim.Circuit,
        alloc: QubitAllocation,
        ctx: "DetectorContext",
    ) -> PhaseResult:
        """
        Emit the transversal gate operations (single phase).
        
        Transversal gates are the simplest case: one phase emitting the
        physical gate to all data qubits.
        
        Parameters
        ----------
        circuit : stim.Circuit
            Circuit to emit gates into.
        alloc : QubitAllocation
            Qubit allocation with block-to-global mappings.
        ctx : DetectorContext
            Detector context - used to propagate Pauli frame through the gate.
            
        Returns
        -------
        PhaseResult
            Gate phase result with stabilizer transform.
        """
        self._current_phase += 1
        
        if self.is_two_qubit:
            # Two-qubit gate between block_0 and block_1
            block0 = alloc.get_block("block_0")
            block1 = alloc.get_block("block_1")
            
            if block0 is None or block1 is None:
                raise ValueError("Two-qubit gate requires block_0 and block_1")
            
            n = min(block0.data_count, block1.data_count)
            
            for i in range(n):
                ctrl = block0.data_start + i
                tgt = block1.data_start + i
                circuit.append(self.stim_gate, [ctrl, tgt])
            
            # Propagate Pauli frame through two-qubit gate
            # This updates the tracker's internal frame for each logical qubit pair
            ctx.propagate_frame_through_gate(self.gate_name, qubit=0, target_qubit=1)
        else:
            # Single-qubit gate on all data qubits of block_0
            block = alloc.get_block("block_0")
            
            if block is None:
                raise ValueError("Single-qubit gate requires block_0")
            
            qubits = list(block.data_range)
            circuit.append(self.stim_gate, qubits)
            
            # Propagate Pauli frame through single-qubit gate
            # This updates how X/Z frame corrections transform through the gate
            ctx.propagate_frame_through_gate(self.gate_name, qubit=0)
        
        circuit.append("TICK")
        
        return PhaseResult.gate_phase(
            is_final=True,
            transform=self.get_stabilizer_transform(),
        )
    
    def get_stabilizer_transform(self) -> StabilizerTransform:
        """
        Return how this gate transforms stabilizer types.
        
        Subclasses should override this method. The base implementation provides
        a fallback using ObservableTransform.from_gate_name().
        
        For transversal gates:
        - H: Swaps X ↔ Z, requires history swap
        - S/S_DAG: X → Y, Z → Z
        - X/Y/Z: No change (Pauli gates)
        - T/T_DAG: No change (not Clifford, but stabilizers unchanged)
        - CNOT/CZ/SWAP: Identity per-block (cross-block effects tracked separately)
        """
        # Fallback: use from_gate_name (subclasses should override)
        return ObservableTransform.from_gate_name(self.gate_name).to_stabilizer_transform()
    
    def get_observable_transform(self) -> ObservableTransform:
        """
        Return how this gate transforms the logical observable.
        
        Subclasses should override this method. The base implementation provides
        a fallback using ObservableTransform.from_gate_name().
        """
        return ObservableTransform.from_gate_name(self.gate_name)
    
    def compute_layout(self, codes: List[Code]) -> GadgetLayout:
        """Compute layout for transversal gate."""
        self._last_codes = codes
        self._dimension = self.infer_dimension(codes)
        
        layout = GadgetLayout(target_dim=self._dimension)
        
        if self.is_two_qubit:
            # Two code blocks, placed side by side with auto_offset
            for i, code in enumerate(codes):
                layout.add_block(
                    name=f"block_{i}",
                    code=code,
                    auto_offset=True,
                    margin=3.0,  # Spacing between blocks
                )
        else:
            # Single code block at origin
            code = codes[0]
            layout.add_block(
                name="block_0",
                code=code,
                offset=(0.0,) * self._dimension,
            )
        
        self._cached_layout = layout
        return layout
    
    
    def _build_metadata(
        self,
        layout: GadgetLayout,
        codes: List[Code],
    ) -> GadgetMetadata:
        """Build metadata for decoder."""
        qubit_map = layout.get_qubit_index_map()
        
        return GadgetMetadata(
            gadget_type="transversal",
            logical_operation=self.gate_name,
            input_codes=[type(c).__name__ for c in codes],
            output_codes=[type(c).__name__ for c in codes],
            qubit_index_map=qubit_map,
            detector_coords=self.get_detector_coords(layout),
            timing_info={
                "rounds_before": self.num_rounds_before,
                "rounds_after": self.num_rounds_after,
                "total_layers": len(self._cached_scheduler.layers) if self._cached_scheduler else 0,
            },
            extra={
                "gate_name": self.gate_name,
                "is_two_qubit": self.is_two_qubit,
            },
        )
    
    def get_metadata(self) -> GadgetMetadata:
        """
        Get metadata for decoder integration.
        
        Note: For transversal gates used with FaultTolerantGadgetExperiment,
        the metadata is built on demand using cached layout/codes from 
        compute_layout(). Returns a basic metadata if not yet computed.
        """
        if self._cached_metadata is not None:
            return self._cached_metadata
        
        # Try to build metadata from cached layout and codes
        if self._cached_layout is not None and self._last_codes is not None:
            self._cached_metadata = self._build_metadata(
                self._cached_layout, 
                self._last_codes
            )
            return self._cached_metadata
        
        # Return minimal metadata if no layout computed yet
        return GadgetMetadata(
            gadget_type="transversal",
            logical_operation=self.gate_name,
            extra={
                "gate_name": self.gate_name,
                "is_two_qubit": self.is_two_qubit,
                "note": "Full metadata available after compute_layout() is called",
            },
        )


# Convenience classes for specific gates

class TransversalHadamard(TransversalGate):
    """Transversal Hadamard gate (single-qubit)."""
    
    def __init__(self, **kwargs):
        super().__init__(gate_name="H", **kwargs)
    
    def get_stabilizer_transform(self) -> StabilizerTransform:
        """Hadamard swaps X ↔ Z stabilizers."""
        return StabilizerTransform.hadamard()
    
    def get_observable_transform(self) -> ObservableTransform:
        """Hadamard swaps X ↔ Z observables."""
        return ObservableTransform.hadamard()


class TransversalS(TransversalGate):
    """Transversal S gate (single-qubit phase gate)."""
    
    def __init__(self, **kwargs):
        super().__init__(gate_name="S", **kwargs)
    
    def get_stabilizer_transform(self) -> StabilizerTransform:
        """S gate: X → Y, Z → Z.
        
        The S gate transforms X stabilizers to Y stabilizers (up to phase).
        This means X stabilizer measurement outcomes are NOT comparable across
        the gate - we must clear history. Z stabilizers are unchanged.
        
        We set clear_history=True but not swap_xz (since Z basis memory
        still uses Z as the deterministic basis).
        """
        return StabilizerTransform(x_becomes="Y", z_becomes="Z", clear_history=True)
    
    def get_observable_transform(self) -> ObservableTransform:
        """S gate: X → Y, Z → Z, Y → -X."""
        return ObservableTransform.s_gate()


class TransversalSDag(TransversalGate):
    """Transversal S† gate."""
    
    def __init__(self, **kwargs):
        super().__init__(gate_name="S_DAG", **kwargs)
    
    def get_stabilizer_transform(self) -> StabilizerTransform:
        """S† gate: X → -Y, Z → Z.
        
        Similar to S, the S† gate transforms X stabilizers, requiring history clear.
        """
        return StabilizerTransform(x_becomes="Y", z_becomes="Z", clear_history=True)
    
    def get_observable_transform(self) -> ObservableTransform:
        """S† gate: X → -Y, Z → Z, Y → X."""
        return ObservableTransform(transform={"X": "-Y", "Z": "Z", "Y": "X"})


class TransversalT(TransversalGate):
    """
    Transversal T gate (single-qubit).
    
    WARNING: The T gate is NOT a Clifford gate and cannot be simulated in Stim.
    Stim only supports stabilizer/Clifford operations.
    
    For T gates in fault-tolerant circuits, use:
    - Magic state distillation + teleportation
    - TeleportedT gadget (which tracks T as a logical operation)
    
    This class is kept for interface compatibility but will raise an error
    if you try to generate a Stim circuit from it.
    """
    
    def __init__(self, **kwargs):
        # Don't call super().__init__ with gate_name since T is not in STIM_GATE_MAP
        # Just initialize the base Gadget
        from qectostim.gadgets.base import Gadget
        Gadget.__init__(self)
        self.gate_name = "T"
        self.stim_gate = None  # Not supported
        self.is_two_qubit = False
        self.include_stabilizer_rounds = kwargs.get('include_stabilizer_rounds', False)
        self.num_rounds_before = kwargs.get('num_rounds_before', 1)
        self.num_rounds_after = kwargs.get('num_rounds_after', 1)
    
    def emit_next_phase(self, circuit, alloc, ctx):
        """T gate cannot be emitted to Stim."""
        raise NotImplementedError(
            "TransversalT cannot be simulated in Stim (T is non-Clifford). "
            "Use TeleportedT for magic state injection or a different simulator."
        )
    
    @property
    def num_phases(self) -> int:
        return 1
    
    def get_stabilizer_transform(self) -> StabilizerTransform:
        """T is not Clifford, but stabilizer types are unchanged."""
        return StabilizerTransform.identity()
    
    def get_observable_transform(self) -> ObservableTransform:
        """T doesn't change Pauli type (just adds phase)."""
        return ObservableTransform.identity()


class TransversalTDag(TransversalGate):
    """
    Transversal T† gate.
    
    WARNING: T† is NOT a Clifford gate and cannot be simulated in Stim.
    See TransversalT for details.
    """
    
    def __init__(self, **kwargs):
        from qectostim.gadgets.base import Gadget
        Gadget.__init__(self)
        self.gate_name = "T_DAG"
        self.stim_gate = None
        self.is_two_qubit = False
        self.include_stabilizer_rounds = kwargs.get('include_stabilizer_rounds', False)
        self.num_rounds_before = kwargs.get('num_rounds_before', 1)
        self.num_rounds_after = kwargs.get('num_rounds_after', 1)
    
    def emit_next_phase(self, circuit, alloc, ctx):
        """T† gate cannot be emitted to Stim."""
        raise NotImplementedError(
            "TransversalTDag cannot be simulated in Stim (T† is non-Clifford). "
            "Use teleportation-based T for magic state injection."
        )
    
    @property
    def num_phases(self) -> int:
        return 1
    
    def get_stabilizer_transform(self) -> StabilizerTransform:
        """T† is not Clifford, but stabilizer types are unchanged."""
        return StabilizerTransform.identity()
    
    def get_observable_transform(self) -> ObservableTransform:
        """T† doesn't change Pauli type (just adds phase)."""
        return ObservableTransform.identity()


class TransversalX(TransversalGate):
    """Transversal X gate (logical bit flip)."""
    
    def __init__(self, **kwargs):
        super().__init__(gate_name="X", **kwargs)
    
    def get_stabilizer_transform(self) -> StabilizerTransform:
        """X commutes with X stabilizers, anti-commutes with Z (but type unchanged)."""
        return StabilizerTransform.pauli()
    
    def get_observable_transform(self) -> ObservableTransform:
        """X: X → X, Z → -Z, Y → -Y."""
        return ObservableTransform(transform={"X": "X", "Z": "-Z", "Y": "-Y"})


class TransversalY(TransversalGate):
    """Transversal Y gate."""
    
    def __init__(self, **kwargs):
        super().__init__(gate_name="Y", **kwargs)
    
    def get_stabilizer_transform(self) -> StabilizerTransform:
        """Y anti-commutes with both X and Z (but types unchanged)."""
        return StabilizerTransform.pauli()
    
    def get_observable_transform(self) -> ObservableTransform:
        """Y: X → -X, Z → -Z, Y → Y."""
        return ObservableTransform(transform={"X": "-X", "Z": "-Z", "Y": "Y"})


class TransversalZ(TransversalGate):
    """Transversal Z gate (logical phase flip)."""
    
    def __init__(self, **kwargs):
        super().__init__(gate_name="Z", **kwargs)
    
    def get_stabilizer_transform(self) -> StabilizerTransform:
        """Z commutes with Z stabilizers, anti-commutes with X (but type unchanged)."""
        return StabilizerTransform.pauli()
    
    def get_observable_transform(self) -> ObservableTransform:
        """Z: X → -X, Z → Z, Y → -Y."""
        return ObservableTransform(transform={"X": "-X", "Z": "Z", "Y": "-Y"})


class TransversalCNOT(TransversalGate):
    """
    Transversal CNOT gate between two code blocks.
    
    The first code acts as control, second as target.
    Applies CNOT(i, j) where i is the i-th qubit of code 1
    and j is the i-th qubit of code 2.
    
    Observable transforms:
        X_ctrl → X_ctrl (unchanged)
        Z_ctrl → Z_ctrl ⊗ Z_tgt (spreads to target)
        X_tgt → X_ctrl ⊗ X_tgt (picks up control)
        Z_tgt → Z_tgt (unchanged)
    """
    
    def __init__(self, **kwargs):
        super().__init__(gate_name="CNOT", **kwargs)
    
    def get_stabilizer_transform(self) -> StabilizerTransform:
        """
        CNOT transforms stabilizers across blocks.
        
        Two-qubit gates entangle the blocks, so measurements from before
        the gate cannot be compared to measurements after. We must clear
        the measurement history to avoid non-deterministic detectors.
        """
        return StabilizerTransform.identity(clear_history=True)
    
    def get_observable_transform(self) -> ObservableTransform:
        """Return simplified single-block transform (identity per block)."""
        return ObservableTransform.identity()
    
    def get_two_qubit_observable_transform(self) -> TwoQubitObservableTransform:
        """
        Full two-qubit observable transform for CNOT.
        
        CNOT (control=block_0, target=block_1):
            X_ctrl → X_ctrl
            Z_ctrl → Z_ctrl ⊗ Z_tgt
            X_tgt → X_ctrl ⊗ X_tgt
            Z_tgt → Z_tgt
        """
        return TwoQubitObservableTransform.cnot()


class TransversalCZ(TransversalGate):
    """
    Transversal CZ gate between two code blocks.
    
    Applies CZ(i, j) between corresponding qubits of the two codes.
    
    Observable transforms (symmetric):
        X_ctrl → X_ctrl ⊗ Z_tgt (picks up Z from target)
        Z_ctrl → Z_ctrl (unchanged)
        X_tgt → Z_ctrl ⊗ X_tgt (picks up Z from control)
        Z_tgt → Z_tgt (unchanged)
    """
    
    def __init__(self, **kwargs):
        super().__init__(gate_name="CZ", **kwargs)
    
    def get_stabilizer_transform(self) -> StabilizerTransform:
        """
        CZ transforms stabilizers across blocks.
        
        Two-qubit gates entangle the blocks, so measurements from before
        the gate cannot be compared to measurements after. We must clear
        the measurement history to avoid non-deterministic detectors.
        """
        return StabilizerTransform.identity(clear_history=True)
    
    def get_observable_transform(self) -> ObservableTransform:
        """Return simplified single-block transform (identity per block)."""
        return ObservableTransform.identity()
    
    def get_two_qubit_observable_transform(self) -> TwoQubitObservableTransform:
        """
        Full two-qubit observable transform for CZ.
        
        CZ (symmetric):
            X_ctrl → X_ctrl ⊗ Z_tgt
            Z_ctrl → Z_ctrl
            X_tgt → Z_ctrl ⊗ X_tgt
            Z_tgt → Z_tgt
        """
        return TwoQubitObservableTransform.cz()


class TransversalSWAP(TransversalGate):
    """
    Transversal SWAP between two code blocks.
    
    Swaps logical states between the two code blocks.
    
    Observable transforms:
        X_ctrl ↔ X_tgt (swapped)
        Z_ctrl ↔ Z_tgt (swapped)
    """
    
    def __init__(self, **kwargs):
        super().__init__(gate_name="SWAP", **kwargs)
    
    def get_stabilizer_transform(self) -> StabilizerTransform:
        """
        SWAP exchanges states between blocks.
        
        Two-qubit gates entangle the blocks, so measurements from before
        the gate cannot be compared to measurements after. We must clear
        the measurement history to avoid non-deterministic detectors.
        """
        return StabilizerTransform.identity(clear_history=True)
    
    def get_observable_transform(self) -> ObservableTransform:
        """Return simplified single-block transform (identity per block)."""
        return ObservableTransform.identity()
    
    def get_two_qubit_observable_transform(self) -> TwoQubitObservableTransform:
        """
        Full two-qubit observable transform for SWAP.
        
        SWAP:
            X_ctrl ↔ X_tgt
            Z_ctrl ↔ Z_tgt
        """
        return TwoQubitObservableTransform.swap()


def get_transversal_gadget(gate_name: str, **kwargs) -> TransversalGate:
    """
    Factory function to get a transversal gadget by gate name.
    
    Args:
        gate_name: Name of gate (H, S, T, CNOT, CZ, etc.)
        **kwargs: Additional arguments passed to gadget constructor
        
    Returns:
        TransversalGate instance for the specified gate
        
    Example:
        >>> gadget = get_transversal_gadget("H")
        >>> gadget = get_transversal_gadget("CNOT", include_stabilizer_rounds=False)
    """
    gate_map = {
        "H": TransversalHadamard,
        "S": TransversalS,
        "S_DAG": TransversalSDag,
        "T": TransversalT,
        "T_DAG": TransversalTDag,
        "X": TransversalX,
        "Y": TransversalY,
        "Z": TransversalZ,
        "CNOT": TransversalCNOT,
        "CX": TransversalCNOT,
        "CZ": TransversalCZ,
        "SWAP": TransversalSWAP,
    }
    
    if gate_name not in gate_map:
        raise ValueError(
            f"Unknown transversal gate '{gate_name}'. "
            f"Supported: {list(gate_map.keys())}"
        )
    
    return gate_map[gate_name](**kwargs)
