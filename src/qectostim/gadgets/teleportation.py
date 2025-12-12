# src/qectostim/gadgets/teleportation.py
"""
Teleportation-based logical gate gadgets.

Teleportation-based gates implement logical Clifford gates using:
1. Preparation of ancilla qubits in specific states
2. Transversal Bell measurements between data and ancilla
3. Classical correction based on measurement outcomes

This approach enables:
- Logical Cliffords on any CSS code
- Gate teleportation for non-transversal gates
- Magic state injection for T gates

The key insight is that the logical Clifford is determined by the
ancilla preparation, not the code structure.

Example usage:
    >>> from qectostim.gadgets.teleportation import TeleportedHadamard, TeleportedS
    >>> from qectostim.codes.surface import SurfaceCode
    >>> from qectostim.noise.models import UniformDepolarizing
    >>>
    >>> code = SurfaceCode(distance=3)
    >>> noise = UniformDepolarizing(p=0.001)
    >>> 
    >>> # Teleported Hadamard
    >>> h_gadget = TeleportedHadamard()
    >>> circuit = h_gadget.to_stim([code], noise)
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
)
from enum import Enum

import stim

from qectostim.codes.abstract_code import Code
from qectostim.noise.models import NoiseModel
from qectostim.gadgets.base import (
    Gadget,
    GadgetMetadata,
    StabilizerTransform,
    ObservableTransform,
    TeleportationGadgetMixin,
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
    BridgeAncilla,
    QubitIndexMap,
    QubitAllocation,
)
from qectostim.gadgets.scheduling import GadgetScheduler, CircuitLayer

# TYPE_CHECKING imports
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from qectostim.experiments.stabilizer_rounds import DetectorContext


class AncillaState(Enum):
    """States for teleportation ancilla preparation."""
    PLUS = "|+⟩"           # |+⟩ = (|0⟩ + |1⟩)/√2
    ZERO = "|0⟩"           # |0⟩
    Y_PLUS = "|+i⟩"        # |+i⟩ = (|0⟩ + i|1⟩)/√2
    BELL = "|Φ+⟩"          # Bell state (|00⟩ + |11⟩)/√2
    MAGIC_T = "|T⟩"        # Magic state for T gate


@dataclass
class TeleportationProtocol:
    """
    Specification of a teleportation protocol for a logical gate.
    
    Attributes:
        gate_name: Name of logical gate being implemented
        ancilla_state: State to prepare ancilla in
        measurement_basis: Basis for measurement ("X", "Z", or "XZ" for Bell)
        correction_map: Dict mapping measurement outcomes to Pauli corrections
        requires_magic_state: Whether this needs a prepared magic state
    """
    gate_name: str
    ancilla_state: AncillaState
    measurement_basis: str
    correction_map: Dict[Tuple[int, ...], str] = field(default_factory=dict)
    requires_magic_state: bool = False


# Predefined protocols for standard Clifford gates
TELEPORTATION_PROTOCOLS = {
    "H": TeleportationProtocol(
        gate_name="H",
        ancilla_state=AncillaState.PLUS,
        measurement_basis="XZ",
        correction_map={
            (0, 0): "I",
            (1, 0): "Z",
            (0, 1): "X",
            (1, 1): "Y",
        },
    ),
    "S": TeleportationProtocol(
        gate_name="S",
        ancilla_state=AncillaState.Y_PLUS,
        measurement_basis="XZ",
        correction_map={
            (0, 0): "I",
            (1, 0): "Z",
            (0, 1): "X",
            (1, 1): "Y",
        },
    ),
    "S_DAG": TeleportationProtocol(
        gate_name="S_DAG",
        ancilla_state=AncillaState.Y_PLUS,
        measurement_basis="XZ",
        correction_map={
            (0, 0): "I",
            (1, 0): "Z",
            (0, 1): "X",
            (1, 1): "Y",
        },
    ),
    "T": TeleportationProtocol(
        gate_name="T",
        ancilla_state=AncillaState.MAGIC_T,
        measurement_basis="XZ",
        correction_map={
            (0, 0): "I",
            (1, 0): "S",  # S correction for T gate
            (0, 1): "X",
            (1, 1): "XS",
        },
        requires_magic_state=True,
    ),
    "IDENTITY": TeleportationProtocol(
        gate_name="I",
        ancilla_state=AncillaState.BELL,
        measurement_basis="XZ",
        correction_map={
            (0, 0): "I",
            (1, 0): "Z",
            (0, 1): "X",
            (1, 1): "Y",
        },
    ),
}


class TeleportedGate(Gadget, TeleportationGadgetMixin):
    """
    Base class for teleportation-based logical gates.
    
    Implements the teleportation protocol:
    1. Prepare ancilla block in appropriate state
    2. Create Bell pairs between data and ancilla
    3. Measure in X and Z bases
    4. Apply Pauli frame correction based on outcomes
    
    The logical gate performed depends on the ancilla preparation state.
    
    Attributes:
        protocol: TeleportationProtocol specifying the gate
        include_stabilizer_rounds: Whether to include syndrome measurements
        num_rounds_before: Stabilizer rounds before teleportation
        num_rounds_after: Stabilizer rounds after teleportation
    """
    
    def __init__(
        self,
        protocol: TeleportationProtocol,
        include_stabilizer_rounds: bool = True,
        num_rounds_before: int = 1,
        num_rounds_after: int = 1,
    ):
        """
        Initialize teleported gate gadget.
        
        Args:
            protocol: TeleportationProtocol for this gate
            include_stabilizer_rounds: Include stabilizer measurements
            num_rounds_before: Stabilizer rounds before teleportation
            num_rounds_after: Stabilizer rounds after teleportation
        """
        super().__init__()
        
        self.protocol = protocol
        self.include_stabilizer_rounds = include_stabilizer_rounds
        self.num_rounds_before = num_rounds_before
        self.num_rounds_after = num_rounds_after
        
        # State tracking
        self._ancilla_block_name = "ancilla_block"
        self._data_block_name = "data_block"
    
    def validate_codes(self, codes: List[Code]) -> None:
        """Validate codes for teleportation."""
        super().validate_codes(codes)
        
        if len(codes) != 1:
            raise ValueError(
                f"Teleported gates operate on single code blocks, got {len(codes)}"
            )
        
        # Check if code is CSS (required for standard teleportation)
        code = codes[0]
        if not (hasattr(code, 'is_css') and code.is_css):
            # Warning but don't fail
            pass
    
    # =========================================================================
    # PRIMARY INTERFACE - Required by FaultTolerantGadgetExperiment
    # =========================================================================
    
    @property
    def num_phases(self) -> int:
        """
        Teleportation has 3 phases:
        Phase 0: Ancilla preparation
        Phase 1: CNOT + H for Bell measurement setup
        Phase 2: Measurements (is_final=True with MEASUREMENT type)
        """
        return 3
    
    def emit_next_phase(
        self,
        circuit: stim.Circuit,
        alloc: QubitAllocation,
        ctx: "DetectorContext",
    ) -> PhaseResult:
        """
        Emit the next phase of the teleportation protocol.
        
        Phase 0: Ancilla preparation (H, S, etc.)
        Phase 1: CNOT + H for Bell measurement setup
        Phase 2: Measurements
        
        Parameters
        ----------
        circuit : stim.Circuit
            Circuit to emit gates into.
        alloc : QubitAllocation
            Qubit allocation with block-to-global mappings.
        ctx : DetectorContext
            Detector context (for measurement tracking).
            
        Returns
        -------
        PhaseResult
            Result describing what was emitted.
        """
        data_block = alloc.get_block("block_0")
        ancilla_block = alloc.get_block("block_1")
        
        if data_block is None:
            data_block = alloc.get_block(self._data_block_name)
            ancilla_block = alloc.get_block(self._ancilla_block_name)
        
        if data_block is None or ancilla_block is None:
            raise ValueError("Teleportation requires data and ancilla blocks")
        
        n = min(data_block.data_count, ancilla_block.data_count)
        data_qubits = list(data_block.data_range)[:n]
        ancilla_qubits = list(ancilla_block.data_range)[:n]
        
        phase = self._current_phase
        self._current_phase += 1
        
        if phase == 0:
            # Phase 0: Ancilla preparation
            if self.protocol.ancilla_state == AncillaState.PLUS:
                circuit.append("H", ancilla_qubits)
            elif self.protocol.ancilla_state == AncillaState.ZERO:
                pass  # Already in |0⟩
            elif self.protocol.ancilla_state == AncillaState.Y_PLUS:
                circuit.append("H", ancilla_qubits)
                circuit.append("S", ancilla_qubits)
            elif self.protocol.ancilla_state == AncillaState.BELL:
                circuit.append("H", ancilla_qubits)
            elif self.protocol.ancilla_state == AncillaState.MAGIC_T:
                # IDEALIZED magic state preparation
                # In reality, magic states are prepared via distillation
                # Stim cannot simulate the T gate (non-Clifford), so we 
                # prepare |+⟩ as a placeholder. The error model for magic
                # state distillation should be applied separately.
                circuit.append("H", ancilla_qubits)
                # NOTE: T gate omitted - Stim only supports Clifford gates
            
            circuit.append("TICK")
            
            return PhaseResult(
                phase_type=PhaseType.PREPARATION,
                is_final=False,
                needs_stabilizer_rounds=0,  # No rounds needed after prep
            )
        
        elif phase == 1:
            # Phase 1: CNOT + H for Bell measurement setup
            for d, a in zip(data_qubits, ancilla_qubits):
                circuit.append("CNOT", [d, a])
            circuit.append("TICK")
            
            circuit.append("H", data_qubits)
            circuit.append("TICK")
            
            return PhaseResult(
                phase_type=PhaseType.GATE,
                is_final=False,
                needs_stabilizer_rounds=1,  # Run stabilizers before measurement
            )
        
        elif phase == 2:
            # Phase 2: Measurements (Bell measurement on data qubits only)
            # In teleportation, we measure the DATA qubits in Bell basis
            # The ANCILLA qubits are the OUTPUT and should NOT be measured here
            # (Pauli corrections would be applied based on measurement outcomes)
            circuit.append("M", data_qubits)
            circuit.append("TICK")
            
            return PhaseResult.measurement_phase(
                count=len(data_qubits),
                qubits=data_qubits,
                blocks=["data_block"],  # Only data block is measured/destroyed
                is_final=True,
            )
        
        else:
            return PhaseResult.complete()
    
    def get_stabilizer_transform(self) -> StabilizerTransform:
        """
        Return how this teleportation transforms stabilizer types.
        
        Teleportation transfers state to ancilla block, so this depends
        on the gate being teleported.
        
        Gates that transform X to Y (like S) require clearing stabilizer
        history because CSS codes only measure X and Z, not Y.
        """
        gate = self.protocol.gate_name.upper()
        
        if gate == "H":
            return StabilizerTransform.hadamard()
        elif gate in ("S", "S_DAG"):
            # S: X→Y, Z→Z. We must clear history because X→Y can't be tracked
            # in CSS stabilizer measurements (we only measure X and Z, not Y)
            return StabilizerTransform(x_becomes="Y", z_becomes="Z", clear_history=True)
        elif gate in ("T", "T_DAG"):
            return StabilizerTransform.identity()
        elif gate in ("I", "IDENTITY"):
            return StabilizerTransform.identity()
        else:
            return StabilizerTransform.identity()
    
    def get_observable_transform(self) -> ObservableTransform:
        """
        Return how this teleportation transforms the logical observable.
        """
        return ObservableTransform.from_gate_name(self.protocol.gate_name)
    
    def get_output_block_name(self) -> str:
        """
        Return the name of the block containing the output logical state.
        
        For teleportation, the logical state is teleported from the data block
        to the ancilla block. After teleportation, the logical observable
        should be measured on the ancilla block, not the original data block.
        
        Returns
        -------
        str
            The block name where the output logical state resides.
        """
        return self._ancilla_block_name
    
    def is_teleportation_gadget(self) -> bool:
        """Return True to indicate this is a teleportation-based gadget."""
        return True
    
    def compute_layout(self, codes: List[Code]) -> GadgetLayout:
        """
        Compute layout for teleportation.
        
        Places data block and ancilla block with spacing.
        """
        self._dimension = self.infer_dimension(codes)
        code = codes[0]
        
        layout = GadgetLayout(target_dim=self._dimension)
        
        # Add data block at origin
        layout.add_block(
            name=self._data_block_name,
            code=code,
            offset=(0.0,) * self._dimension,
        )
        
        # Add ancilla block (copy of code structure) offset in first dimension
        # Compute offset based on bounding box
        if hasattr(code, 'qubit_coords'):
            coords = code.qubit_coords()
            if coords:
                # qubit_coords() returns List[Tuple], not Dict
                _, max_corner = get_bounding_box(coords)
                offset_dist = max_corner[0] + 3.0  # Spacing
            else:
                offset_dist = code.n + 3.0
        else:
            offset_dist = code.n + 3.0
        
        ancilla_offset = (offset_dist,) + (0.0,) * (self._dimension - 1)
        layout.add_block(
            name=self._ancilla_block_name,
            code=code,
            offset=ancilla_offset,
        )
        
        self._cached_layout = layout
        return layout
    
    def build_schedule(
        self,
        layout: GadgetLayout,
        codes: List[Code],
        noise_model: Optional[NoiseModel] = None,
    ) -> GadgetScheduler:
        """
        Build schedule for teleportation protocol.
        
        Layers:
        1. Pre-stabilizer rounds on data block
        2. Ancilla preparation
        3. CNOT between data and ancilla (Bell measurement prep)
        4. X measurement on data, Z measurement on ancilla
        5. Post-stabilizer rounds (with Pauli frame update)
        """
        scheduler = GadgetScheduler(layout)
        code = codes[0]
        
        data_block = layout.blocks[self._data_block_name]
        ancilla_block = layout.blocks[self._ancilla_block_name]
        
        n = code.n
        time = 0.0
        
        # Pre-stabilizer rounds
        if self.include_stabilizer_rounds:
            for _ in range(self.num_rounds_before):
                scheduler.schedule_stabilizer_round(
                    code=code,
                    data_offset=data_block.data_qubit_range[0],
                    ancilla_offset=data_block.ancilla_qubit_range[0],
                    time=time,
                )
                time += 1.0
        
        # Ancilla preparation layer
        layer = CircuitLayer(time=time)
        ancilla_start = ancilla_block.data_qubit_range[0]
        
        if self.protocol.ancilla_state == AncillaState.PLUS:
            # Prepare |+⟩: Reset then H
            layer.resets.extend(range(ancilla_start, ancilla_start + n))
            scheduler.layers.append(layer)
            time += 1.0
            
            layer = CircuitLayer(time=time)
            layer.single_qubit_gates.extend([
                ("H", ancilla_start + i) for i in range(n)
            ])
            scheduler.layers.append(layer)
            
        elif self.protocol.ancilla_state == AncillaState.ZERO:
            # Prepare |0⟩: Just reset
            layer.resets.extend(range(ancilla_start, ancilla_start + n))
            scheduler.layers.append(layer)
            
        elif self.protocol.ancilla_state == AncillaState.Y_PLUS:
            # Prepare |+i⟩: Reset, H, S
            layer.resets.extend(range(ancilla_start, ancilla_start + n))
            scheduler.layers.append(layer)
            time += 1.0
            
            layer = CircuitLayer(time=time)
            layer.single_qubit_gates.extend([
                ("H", ancilla_start + i) for i in range(n)
            ])
            scheduler.layers.append(layer)
            time += 1.0
            
            layer = CircuitLayer(time=time)
            layer.single_qubit_gates.extend([
                ("S", ancilla_start + i) for i in range(n)
            ])
            scheduler.layers.append(layer)
            
        elif self.protocol.ancilla_state == AncillaState.BELL:
            # Prepare Bell pair: Reset both, H on ancilla, CNOT
            layer.resets.extend(range(ancilla_start, ancilla_start + n))
            scheduler.layers.append(layer)
            time += 1.0
            
            layer = CircuitLayer(time=time)
            layer.single_qubit_gates.extend([
                ("H", ancilla_start + i) for i in range(n)
            ])
            scheduler.layers.append(layer)
            
        elif self.protocol.ancilla_state == AncillaState.MAGIC_T:
            # Magic state injection - placeholder
            # In practice, this requires distillation
            layer.resets.extend(range(ancilla_start, ancilla_start + n))
            scheduler.layers.append(layer)
            time += 1.0
            
            # Approximate |T⟩ state prep
            layer = CircuitLayer(time=time)
            layer.single_qubit_gates.extend([
                ("H", ancilla_start + i) for i in range(n)
            ])
            scheduler.layers.append(layer)
            time += 1.0
            
            layer = CircuitLayer(time=time)
            layer.single_qubit_gates.extend([
                ("T", ancilla_start + i) for i in range(n)
            ])
            scheduler.layers.append(layer)
        
        time += 1.0
        
        # CNOT between data and ancilla (Bell measurement preparation)
        layer = CircuitLayer(time=time)
        data_start = data_block.data_qubit_range[0]
        layer.two_qubit_gates.extend([
            ("CNOT", data_start + i, ancilla_start + i)
            for i in range(n)
        ])
        scheduler.layers.append(layer)
        time += 1.0
        
        # Hadamard on data qubits (for X measurement)
        layer = CircuitLayer(time=time)
        layer.single_qubit_gates.extend([
            ("H", data_start + i) for i in range(n)
        ])
        scheduler.layers.append(layer)
        time += 1.0
        
        # Measurements
        layer = CircuitLayer(time=time)
        # Measure data (X basis, after H)
        layer.measurements.extend([(data_start + i, "Z") for i in range(n)])
        # Measure ancilla (Z basis)
        layer.measurements.extend([(ancilla_start + i, "Z") for i in range(n)])
        scheduler.layers.append(layer)
        time += 1.0
        
        # Post-stabilizer rounds on remaining qubits
        # (In practice, the output is now on the ancilla side)
        if self.include_stabilizer_rounds:
            for _ in range(self.num_rounds_after):
                scheduler.schedule_stabilizer_round(
                    code=code,
                    data_offset=ancilla_block.data_qubit_range[0],
                    ancilla_offset=ancilla_block.ancilla_qubit_range[0],
                    time=time,
                )
                time += 1.0
        
        self._cached_scheduler = scheduler
        return scheduler
    
    def _add_detectors(
        self,
        circuit: stim.Circuit,
        layout: GadgetLayout,
        codes: List[Code],
    ) -> None:
        """Add detector instructions."""
        # Placeholder - full implementation tracks stabilizers
        pass
    
    def _add_logical_observable(
        self,
        circuit: stim.Circuit,
        layout: GadgetLayout,
        codes: List[Code],
    ) -> None:
        """Add logical observable tracking."""
        # Placeholder - tracks logical operator through teleportation
        pass
    
    def _build_metadata(
        self,
        layout: GadgetLayout,
        codes: List[Code],
    ) -> GadgetMetadata:
        """Build metadata for decoder."""
        qubit_map = layout.get_qubit_index_map()
        
        return GadgetMetadata(
            gadget_type="teleportation",
            logical_operation=self.protocol.gate_name,
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
                "protocol": self.protocol.gate_name,
                "ancilla_state": self.protocol.ancilla_state.value,
                "requires_magic_state": self.protocol.requires_magic_state,
            },
        )
    
    def get_metadata(self) -> GadgetMetadata:
        """Get cached metadata."""
        if self._cached_metadata is None:
            raise RuntimeError("Metadata not available. Call to_stim() first.")
        return self._cached_metadata


# Convenience classes for specific gates

class TeleportedHadamard(TeleportedGate):
    """Teleported Hadamard gate."""
    
    def __init__(self, **kwargs):
        super().__init__(protocol=TELEPORTATION_PROTOCOLS["H"], **kwargs)


class TeleportedS(TeleportedGate):
    """Teleported S (phase) gate."""
    
    def __init__(self, **kwargs):
        super().__init__(protocol=TELEPORTATION_PROTOCOLS["S"], **kwargs)


class TeleportedSDag(TeleportedGate):
    """Teleported S† gate."""
    
    def __init__(self, **kwargs):
        super().__init__(protocol=TELEPORTATION_PROTOCOLS["S_DAG"], **kwargs)


class TeleportedT(TeleportedGate):
    """
    Teleported T gate using magic state injection.
    
    Note: This requires a pre-prepared magic state. In a full
    implementation, the magic state would come from distillation.
    Here we use an idealized preparation.
    """
    
    def __init__(self, **kwargs):
        super().__init__(protocol=TELEPORTATION_PROTOCOLS["T"], **kwargs)


class TeleportedIdentity(TeleportedGate):
    """
    Teleported identity (for testing or as building block).
    
    Implements logical identity via teleportation, useful for:
    - Testing teleportation infrastructure
    - Logical state transfer between code blocks
    """
    
    def __init__(self, **kwargs):
        super().__init__(protocol=TELEPORTATION_PROTOCOLS["IDENTITY"], **kwargs)


def get_teleported_gadget(gate_name: str, **kwargs) -> TeleportedGate:
    """
    Factory function to get a teleported gadget by gate name.
    
    Args:
        gate_name: Name of gate (H, S, S_DAG, T, I/IDENTITY)
        **kwargs: Additional arguments for gadget
        
    Returns:
        TeleportedGate instance
    """
    gate_map = {
        "H": TeleportedHadamard,
        "S": TeleportedS,
        "S_DAG": TeleportedSDag,
        "T": TeleportedT,
        "I": TeleportedIdentity,
        "IDENTITY": TeleportedIdentity,
    }
    
    if gate_name not in gate_map:
        raise ValueError(
            f"Unknown teleported gate '{gate_name}'. "
            f"Supported: {list(gate_map.keys())}"
        )
    
    return gate_map[gate_name](**kwargs)
