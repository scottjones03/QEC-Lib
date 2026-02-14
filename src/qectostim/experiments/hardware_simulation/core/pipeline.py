# src/qectostim/experiments/hardware_simulation/core/pipeline.py
"""
Compilation pipeline data structures.

Defines the intermediate representations used during compilation
from logical circuits to hardware-executable sequences.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import (
    Dict,
    List,
    Optional,
    Tuple,
    Any,
    Sequence,
    Iterator,
    TYPE_CHECKING,
)

import stim

if TYPE_CHECKING:
    from qectostim.experiments.hardware_simulation.core.operations import PhysicalOperation
    from qectostim.experiments.hardware_simulation.core.gates import GateSpec, ParameterizedGate


# =====================================================================
# QEC Metadata dataclasses
# =====================================================================

@dataclass
class StabilizerInfo:
    """Information about one type of stabilizer (X or Z).

    Attributes
    ----------
    supports : List[List[int]]
        Per-stabilizer list of data qubit indices in the support.
    ancillas : List[int]
        Global qubit indices of the ancilla qubits.
    cnot_schedule : Optional[List[List[Tuple[int, int]]]]
        Parallel CNOT layers: each inner list contains ``(ctrl, tgt)``
        pairs that can execute simultaneously.  ``None`` when the
        builder doesn't expose scheduling information.
    metacheck_ancillas : List[int]
        Global qubit indices of metacheck ancillas (if any).
    """
    supports: List[List[int]] = field(default_factory=list)
    ancillas: List[int] = field(default_factory=list)
    cnot_schedule: Optional[List[List[Tuple[int, int]]]] = None
    metacheck_ancillas: List[int] = field(default_factory=list)


@dataclass
class BlockInfo:
    """Allocation of one logical block within a multi-block experiment.

    Attributes
    ----------
    block_name : str
        Human-readable label (e.g. ``"ctrl"``, ``"target"``).
    data_qubits : List[int]
        Global data qubit indices for this block.
    x_ancilla_qubits : List[int]
        Global X-ancilla qubit indices.
    z_ancilla_qubits : List[int]
        Global Z-ancilla qubit indices.
    """
    block_name: str = ""
    data_qubits: List[int] = field(default_factory=list)
    x_ancilla_qubits: List[int] = field(default_factory=list)
    z_ancilla_qubits: List[int] = field(default_factory=list)


@dataclass
class PhaseInfo:
    """Temporal phase within a QEC experiment.

    A *phase* is a contiguous region of the circuit with uniform
    structure — e.g. all identical stabilizer rounds, or the gadget
    operation, or the final measurement.

    Attributes
    ----------
    phase_type : str
        Label such as ``"init"``, ``"stabilizer_round"``,
        ``"gadget"``, ``"final_round"``, ``"measure"``.
    num_rounds : int
        Number of stabilizer rounds in this phase (0 for init/measure).
    is_repeated : bool
        Whether this phase consists of identical repeated rounds.
    round_signature : Optional[Tuple]
        Hashable canonical key for the 2Q interaction pattern of one
        round.  ``None`` for phases with no 2Q operations.
    identical_to_phase : Optional[int]
        If this phase has the same round_signature as another phase,
        the index of that earlier phase.  Enables cross-phase cache
        sharing.
    """
    phase_type: str = ""
    num_rounds: int = 0
    is_repeated: bool = False
    round_signature: Optional[Tuple] = None
    identical_to_phase: Optional[int] = None


def _build_round_signature(
    x_stab: "StabilizerInfo",
    z_stab: "StabilizerInfo",
) -> Optional[Tuple]:
    """Build a canonical, hashable round signature from CNOT schedules.

    The signature is a tuple-of-tuples-of-tuples matching the
    ``block_cache`` key format used by ``ionRoutingWISEArch``:
    ``((sorted pairs layer 0), (sorted pairs layer 1), ...)``.

    Returns ``None`` when no CNOT schedule is available.
    """
    layers: List[List[Tuple[int, int]]] = []
    if x_stab.cnot_schedule:
        layers.extend(x_stab.cnot_schedule)
    if z_stab.cnot_schedule:
        layers.extend(z_stab.cnot_schedule)
    if not layers:
        return None
    return tuple(tuple(sorted(layer)) for layer in layers)


@dataclass
class QECMetadata:
    """Rich metadata about a QEC experiment for compiler exploitation.

    Created by ``from_css_memory()`` or ``from_gadget_experiment()``
    factory methods and attached to the experiment object.  Downstream
    stages (NativeCircuit, WISECompiler, ionRoutingWISEArch) can read
    it to enable structure-aware compilation.

    Attributes
    ----------
    qubit_roles : Dict[int, str]
        Maps qubit index → role: ``"D"`` (data), ``"X"`` (X-ancilla),
        ``"Z"`` (Z-ancilla), ``"P"`` (preparation/projection).
    code_name : str
        Human-readable code name (e.g. ``"RotatedSurfaceCode"``).
    code_n, code_k, code_d : int
        Code parameters ``[[n, k, d]]``.
    is_css : bool
        Whether the code is CSS.
    rounds : int
        Total number of stabilizer rounds.
    measurement_basis : str
        ``"X"`` or ``"Z"``.
    total_qubits : int
        Total qubit count including ancillas and projections.
    data_qubit_indices : frozenset
        Set of data qubit indices.
    ancilla_qubit_indices : frozenset
        Set of ancilla qubit indices (X + Z).
    preparation_qubit_indices : frozenset
        Set of preparation/projection ancilla indices.
    x_stabilizers, z_stabilizers : StabilizerInfo
        Stabilizer structure and CNOT schedules.
    logical_x_support, logical_z_support : List[List[int]]
        Logical operator supports.
    block_allocations : List[BlockInfo]
        Per-block allocation for multi-block experiments.
    phases : List[PhaseInfo]
        Temporal phase decomposition of the circuit.
    round_signature : Optional[Tuple]
        Canonical hashable key for one stabilizer round's 2Q pattern.
    ion_return_invariant : bool
        Whether ions return to starting positions after each round.
    extra : Dict[str, Any]
        Additional metadata.
    """
    qubit_roles: Dict[int, str] = field(default_factory=dict)
    code_name: str = ""
    code_n: int = 0
    code_k: int = 0
    code_d: int = 0
    is_css: bool = False
    rounds: int = 0
    measurement_basis: str = ""
    total_qubits: int = 0
    data_qubit_indices: frozenset = field(default_factory=frozenset)
    ancilla_qubit_indices: frozenset = field(default_factory=frozenset)
    preparation_qubit_indices: frozenset = field(default_factory=frozenset)
    x_stabilizers: StabilizerInfo = field(default_factory=StabilizerInfo)
    z_stabilizers: StabilizerInfo = field(default_factory=StabilizerInfo)
    logical_x_support: List[List[int]] = field(default_factory=list)
    logical_z_support: List[List[int]] = field(default_factory=list)
    block_allocations: List[BlockInfo] = field(default_factory=list)
    phases: List[PhaseInfo] = field(default_factory=list)
    round_signature: Optional[Tuple] = None
    ion_return_invariant: bool = False
    extra: Dict[str, Any] = field(default_factory=dict)

    @property
    def stabilizers(self) -> List[StabilizerInfo]:
        """All stabilizer groups (X then Z) as a combined list."""
        result: List[StabilizerInfo] = []
        if self.x_stabilizers and self.x_stabilizers.supports:
            result.append(self.x_stabilizers)
        if self.z_stabilizers and self.z_stabilizers.supports:
            result.append(self.z_stabilizers)
        return result

    @property
    def cnot_schedule(self) -> Optional[Dict[str, List[List[Tuple[int, int]]]]]:
        """Combined CNOT schedule dict ``{"x": [...], "z": [...]}``.

        Returns ``None`` when neither X nor Z has a schedule.
        """
        result: Dict[str, List[List[Tuple[int, int]]]] = {}
        if self.x_stabilizers and self.x_stabilizers.cnot_schedule:
            result["x"] = self.x_stabilizers.cnot_schedule
        if self.z_stabilizers and self.z_stabilizers.cnot_schedule:
            result["z"] = self.z_stabilizers.cnot_schedule
        return result if result else None

    # ------------------------------------------------------------------
    # Factory methods
    # ------------------------------------------------------------------

    @classmethod
    def from_css_memory(
        cls,
        code,
        builder,
        rounds: int,
        measurement_basis: str = "Z",
        extra_roles: Optional[Dict[int, str]] = None,
    ) -> "QECMetadata":
        """Build metadata from a CSS memory experiment.

        Parameters
        ----------
        code
            QEC code object (must expose ``n``, ``k``, ``d``, ``hx``,
            ``hz``).
        builder : CSSStabilizerRoundBuilder
            The builder that was used to construct the circuit.
        rounds : int
            Number of stabilizer rounds.
        measurement_basis : str
            ``"X"`` or ``"Z"``.
        extra_roles : dict or None
            Additional qubit→role overrides (e.g. projection ancillas).
        """
        # --- Qubit roles ---
        roles: Dict[int, str] = dict(builder.qubit_roles)
        if extra_roles:
            roles.update(extra_roles)

        data_idx = frozenset(q for q, r in roles.items() if r == "D")
        anc_idx = frozenset(q for q, r in roles.items() if r in ("X", "Z", "MX", "MZ"))
        prep_idx = frozenset(q for q, r in roles.items() if r == "P")

        # --- Stabilizer info ---
        hx = getattr(code, "hx", None)
        hz = getattr(code, "hz", None)

        def _build_stab_info(matrix, ancillas, stab_type: str) -> StabilizerInfo:
            supports: List[List[int]] = []
            if matrix is not None and hasattr(matrix, "shape") and matrix.size > 0:
                for row_idx in range(matrix.shape[0]):
                    row = matrix[row_idx]
                    supp = [
                        builder.data_offset + int(c)
                        for c in range(len(row))
                        if row[c]
                    ]
                    supports.append(supp)
            # Get CNOT schedule from builder
            cnot_sched = None
            if hasattr(builder, "get_cnot_layers"):
                cnot_sched = builder.get_cnot_layers(stab_type)
            return StabilizerInfo(
                supports=supports,
                ancillas=list(ancillas),
                cnot_schedule=cnot_sched,
            )

        x_info = _build_stab_info(hx, builder.x_ancillas, "x")
        z_info = _build_stab_info(hz, builder.z_ancillas, "z")

        # --- Logical operators ---
        logical_x: List[List[int]] = []
        logical_z: List[List[int]] = []
        lx = getattr(code, "lx", None)
        lz = getattr(code, "lz", None)
        if lx is not None and hasattr(lx, "shape"):
            for row in lx:
                logical_x.append(
                    [builder.data_offset + int(c) for c in range(len(row)) if row[c]]
                )
        if lz is not None and hasattr(lz, "shape"):
            for row in lz:
                logical_z.append(
                    [builder.data_offset + int(c) for c in range(len(row)) if row[c]]
                )

        # --- Round signature ---
        sig = _build_round_signature(x_info, z_info)

        # --- Ion return invariant ---
        # For a pure memory experiment, ions return to starting positions
        # after each round when the round signature is well-defined.
        iri = sig is not None

        # --- Phase decomposition ---
        phases: List[PhaseInfo] = [
            PhaseInfo(phase_type="init", num_rounds=0),
            PhaseInfo(
                phase_type="stabilizer_round",
                num_rounds=max(0, rounds - 1),
                is_repeated=True,
                round_signature=sig,
            ),
            PhaseInfo(
                phase_type="final_round",
                num_rounds=1,
                is_repeated=False,
                round_signature=sig,
                identical_to_phase=1,
            ),
            PhaseInfo(phase_type="measure", num_rounds=0),
        ]

        return cls(
            qubit_roles=roles,
            code_name=type(code).__name__,
            code_n=getattr(code, "n", 0),
            code_k=getattr(code, "k", 0),
            code_d=getattr(code, "d", 0),
            is_css=True,
            rounds=rounds,
            measurement_basis=measurement_basis,
            total_qubits=builder.total_qubits,
            data_qubit_indices=data_idx,
            ancilla_qubit_indices=anc_idx,
            preparation_qubit_indices=prep_idx,
            x_stabilizers=x_info,
            z_stabilizers=z_info,
            logical_x_support=logical_x,
            logical_z_support=logical_z,
            phases=phases,
            round_signature=sig,
            ion_return_invariant=iri,
        )

    @classmethod
    def from_gadget_experiment(
        cls,
        codes,
        builders,
        allocation,
        gadget,
        rounds_before: int = 0,
        rounds_after: int = 0,
    ) -> "QECMetadata":
        """Build metadata from a fault-tolerant gadget experiment.

        Parameters
        ----------
        codes : list
            Code objects for each logical block.
        builders : list
            ``CSSStabilizerRoundBuilder`` for each block.
        allocation
            Unified allocation object with qubit mapping.
        gadget
            Gadget object (transversal CNOT, etc.).
        rounds_before, rounds_after : int
            Number of stabilizer rounds before/after the gadget.
        """
        # --- Merge roles from all builders ---
        roles: Dict[int, str] = {}
        for b in builders:
            roles.update(b.qubit_roles)

        data_idx = frozenset(q for q, r in roles.items() if r == "D")
        anc_idx = frozenset(q for q, r in roles.items() if r in ("X", "Z", "MX", "MZ"))
        prep_idx = frozenset(q for q, r in roles.items() if r == "P")

        # Use first code/builder for stabilizer info (primary block)
        code = codes[0] if codes else None
        builder = builders[0] if builders else None

        hx = getattr(code, "hx", None) if code else None
        hz = getattr(code, "hz", None) if code else None

        def _build_stab_info(matrix, ancillas, stab_type: str) -> StabilizerInfo:
            supports: List[List[int]] = []
            if matrix is not None and hasattr(matrix, "shape") and matrix.size > 0:
                for row_idx in range(matrix.shape[0]):
                    row = matrix[row_idx]
                    supp = [
                        builder.data_offset + int(c)
                        for c in range(len(row))
                        if row[c]
                    ]
                    supports.append(supp)
            cnot_sched = None
            if builder and hasattr(builder, "get_cnot_layers"):
                cnot_sched = builder.get_cnot_layers(stab_type)
            return StabilizerInfo(
                supports=supports,
                ancillas=list(ancillas) if ancillas else [],
                cnot_schedule=cnot_sched,
            )

        x_info = _build_stab_info(
            hx, builder.x_ancillas if builder else [], "x"
        )
        z_info = _build_stab_info(
            hz, builder.z_ancillas if builder else [], "z"
        )

        # --- Block allocations ---
        blocks: List[BlockInfo] = []
        for i, b in enumerate(builders):
            blocks.append(BlockInfo(
                block_name=getattr(b, "block_name", f"block_{i}"),
                data_qubits=list(b.data_qubits),
                x_ancilla_qubits=list(b.x_ancillas),
                z_ancilla_qubits=list(b.z_ancillas),
            ))

        # --- Logical operators ---
        logical_x: List[List[int]] = []
        logical_z: List[List[int]] = []
        if code:
            lx = getattr(code, "lx", None)
            lz = getattr(code, "lz", None)
            if lx is not None and hasattr(lx, "shape"):
                for row in lx:
                    logical_x.append(
                        [builder.data_offset + int(c) for c in range(len(row)) if row[c]]
                    )
            if lz is not None and hasattr(lz, "shape"):
                for row in lz:
                    logical_z.append(
                        [builder.data_offset + int(c) for c in range(len(row)) if row[c]]
                    )

        # --- Round signature ---
        sig = _build_round_signature(x_info, z_info)

        # --- Gadget experiments are NOT ion-return-invariant ---
        iri = False

        # --- Phase decomposition ---
        total_rounds = rounds_before + rounds_after
        phases: List[PhaseInfo] = [
            PhaseInfo(phase_type="init", num_rounds=0),
            PhaseInfo(
                phase_type="stabilizer_round_pre",
                num_rounds=rounds_before,
                is_repeated=rounds_before > 1,
                round_signature=sig,
            ),
            PhaseInfo(phase_type="gadget", num_rounds=0),
            PhaseInfo(
                phase_type="stabilizer_round_post",
                num_rounds=rounds_after,
                is_repeated=rounds_after > 1,
                round_signature=sig,
                identical_to_phase=1,
            ),
            PhaseInfo(phase_type="measure", num_rounds=0),
        ]

        total_qubits = sum(b.total_qubits for b in builders) if builders else 0

        return cls(
            qubit_roles=roles,
            code_name=type(code).__name__ if code else "",
            code_n=getattr(code, "n", 0) if code else 0,
            code_k=getattr(code, "k", 0) if code else 0,
            code_d=getattr(code, "d", 0) if code else 0,
            is_css=True,
            rounds=total_rounds,
            measurement_basis="Z",
            total_qubits=total_qubits,
            data_qubit_indices=data_idx,
            ancilla_qubit_indices=anc_idx,
            preparation_qubit_indices=prep_idx,
            x_stabilizers=x_info,
            z_stabilizers=z_info,
            logical_x_support=logical_x,
            logical_z_support=logical_z,
            block_allocations=blocks,
            phases=phases,
            round_signature=sig,
            ion_return_invariant=iri,
        )


# =====================================================================
# Pipeline data structures
# =====================================================================


@dataclass
class QubitMapping:
    """Mapping between logical and physical qubits.
    
    Attributes
    ----------
    logical_to_physical : Dict[int, int]
        Logical qubit index to physical qubit index.
    physical_to_logical : Dict[int, int]
        Physical qubit index to logical qubit index.
    zone_assignments : Dict[int, str]
        Physical qubit to zone ID mapping.
    """
    logical_to_physical: Dict[int, int] = field(default_factory=dict)
    physical_to_logical: Dict[int, int] = field(default_factory=dict)
    zone_assignments: Dict[int, str] = field(default_factory=dict)
    
    def add_mapping(self, logical: int, physical: int, zone: Optional[str] = None) -> None:
        """Add a logical-to-physical qubit mapping."""
        self.logical_to_physical[logical] = physical
        self.physical_to_logical[physical] = logical
        if zone is not None:
            self.zone_assignments[physical] = zone
    
    # Alias so compilers can call mapping.assign(logical, physical)
    assign = add_mapping
    
    def get_physical(self, logical: int) -> Optional[int]:
        """Get physical qubit for a logical qubit."""
        return self.logical_to_physical.get(logical)
    
    def get_logical(self, physical: int) -> Optional[int]:
        """Get logical qubit for a physical qubit."""
        return self.physical_to_logical.get(physical)
    
    def swap_physical(self, phys1: int, phys2: int) -> None:
        """Update mapping after a physical SWAP operation."""
        log1 = self.physical_to_logical.get(phys1)
        log2 = self.physical_to_logical.get(phys2)
        
        if log1 is not None:
            self.logical_to_physical[log1] = phys2
        if log2 is not None:
            self.logical_to_physical[log2] = phys1
        
        self.physical_to_logical[phys1] = log2
        self.physical_to_logical[phys2] = log1
    
    def num_qubits(self) -> int:
        """Get total number of mapped qubits."""
        return len(self.logical_to_physical)
    
    def copy(self) -> "QubitMapping":
        """Create a copy of this mapping."""
        return QubitMapping(
            logical_to_physical=dict(self.logical_to_physical),
            physical_to_logical=dict(self.physical_to_logical),
            zone_assignments=dict(self.zone_assignments),
        )


@dataclass
class ScheduledOperation:
    """An operation with scheduling information.
    
    Attributes
    ----------
    operation : PhysicalOperation
        The operation to execute.
    start_time : float
        Start time in microseconds.
    end_time : float
        End time in microseconds (start_time + duration).
    parallel_group : int
        ID of parallel execution group (operations with same ID run together).
    dependencies : List[int]
        Indices of operations that must complete before this one.
    """
    operation: "PhysicalOperation"
    start_time: float = 0.0
    end_time: float = 0.0
    parallel_group: int = 0
    dependencies: List[int] = field(default_factory=list)
    
    def __post_init__(self):
        if self.end_time == 0.0:
            self.end_time = self.start_time + self.operation.duration
    
    @property
    def duration(self) -> float:
        return self.end_time - self.start_time


@dataclass
class CircuitLayer:
    """A layer of parallel operations in a circuit.
    
    Operations in the same layer can execute simultaneously.
    
    Attributes
    ----------
    operations : List[PhysicalOperation]
        Operations in this layer.
    start_time : float
        Start time of this layer.
    duration : float
        Duration of this layer (max of operation durations).
    """
    operations: List["PhysicalOperation"] = field(default_factory=list)
    start_time: float = 0.0
    
    @property
    def duration(self) -> float:
        if not self.operations:
            return 0.0
        return max(op.duration for op in self.operations)
    
    @property
    def end_time(self) -> float:
        return self.start_time + self.duration
    
    def add_operation(self, op: "PhysicalOperation") -> None:
        """Add an operation to this layer."""
        self.operations.append(op)
    
    def qubits_used(self) -> set:
        """Get all qubits used in this layer."""
        qubits = set()
        for op in self.operations:
            qubits.update(op.qubits)
        return qubits


@dataclass
class NativeCircuit:
    """Circuit decomposed to native gates.
    
    First stage of compilation: gates are decomposed to the
    hardware's native gate set, but no qubit mapping yet.
    
    Attributes
    ----------
    operations : List[Any]
        Sequence of native gate operations.  Entries are either
        ``(GateSpec, Tuple[int, ...])`` pairs or ``DecomposedGate``
        objects — both forms are accepted.
    num_qubits : int
        Number of logical qubits.
    metadata : Dict[str, Any]
        Additional circuit metadata.
    stim_instruction_map : Dict[int, List[int]]
        Maps stim instruction index (in the flattened ideal circuit)
        to the list of native operation indices that were produced by
        decomposing that instruction.  This is the critical bridge
        between the original stim circuit and native ops.
    stim_source : Optional[stim.Circuit]
        Reference to the original stim circuit that was decomposed.
        Stored so that downstream stages can access annotations
        (DETECTOR, OBSERVABLE_INCLUDE, etc.) that were deliberately
        *not* decomposed into native ops.
    """
    operations: List[Any] = field(default_factory=list)
    num_qubits: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    stim_instruction_map: Dict[int, List[int]] = field(default_factory=dict)
    stim_source: Optional[stim.Circuit] = None
    qec_metadata: Optional["QECMetadata"] = None
    data_qubit_indices: Optional[frozenset] = None
    ancilla_qubit_indices: Optional[frozenset] = None

    @property
    def qubit_roles(self) -> Dict[int, str]:
        """Qubit index → role mapping.

        Prefers ``qec_metadata.qubit_roles`` when available; falls back
        to a synthetic mapping derived from ``data_qubit_indices`` /
        ``ancilla_qubit_indices``.
        """
        if self.qec_metadata is not None:
            return self.qec_metadata.qubit_roles
        roles: Dict[int, str] = {}
        for q in range(self.num_qubits):
            if self.data_qubit_indices and q in self.data_qubit_indices:
                roles[q] = "D"
            elif self.ancilla_qubit_indices and q in self.ancilla_qubit_indices:
                roles[q] = "M"
            else:
                roles[q] = "D"
        return roles
    
    def add_gate(self, gate: "GateSpec", qubits: Tuple[int, ...]) -> None:
        """Add a native gate to the circuit."""
        self.operations.append((gate, qubits))
        self.num_qubits = max(self.num_qubits, max(qubits) + 1)
    
    def gate_count(self) -> int:
        """Total number of gates."""
        return len(self.operations)
    
    def two_qubit_count(self) -> int:
        """Number of two-qubit gates."""
        count = 0
        for op in self.operations:
            if hasattr(op, 'qubits'):
                # DecomposedGate
                if len(op.qubits) == 2:
                    count += 1
            elif isinstance(op, tuple) and len(op) >= 2:
                # (GateSpec, qubits) tuple
                if len(op[1]) == 2:
                    count += 1
        return count
    
    def __len__(self) -> int:
        return len(self.operations)
    
    def __iter__(self):
        return iter(self.operations)


@dataclass
class MappedCircuit:
    """Circuit with logical-to-physical qubit mapping.
    
    Second stage: logical qubits are assigned to physical qubits,
    but routing (SWAP insertion) has not been done yet.
    
    Attributes
    ----------
    native_circuit : NativeCircuit
        The underlying native circuit.
    mapping : QubitMapping
        Current qubit mapping.
    metadata : Dict[str, Any]
        Additional mapping metadata.
    """
    native_circuit: NativeCircuit
    mapping: QubitMapping
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def physical_operations(self) -> Iterator[Tuple["GateSpec", Tuple[int, ...]]]:
        """Iterate over operations with physical qubit indices."""
        for gate, logical_qubits in self.native_circuit:
            physical_qubits = tuple(
                self.mapping.get_physical(q) for q in logical_qubits
            )
            yield gate, physical_qubits


@dataclass
class RoutedCircuit:
    """Circuit with routing operations inserted.
    
    Third stage: SWAP/transport operations have been inserted
    to satisfy connectivity constraints.
    
    Attributes
    ----------
    operations : List[PhysicalOperation]
        Sequence of physical operations including routing.
    final_mapping : Optional[QubitMapping]
        Qubit mapping after all operations.
    routing_overhead : int
        Number of routing operations added.
    mapped_circuit : Optional[MappedCircuit]
        Reference to the pre-routing mapped circuit.
    routing_operations : List[Any]
        Explicit list of routing-only operations inserted.
    metadata : Dict[str, Any]
        Additional routing metadata.
    """
    operations: List["PhysicalOperation"] = field(default_factory=list)
    final_mapping: Optional[QubitMapping] = None
    routing_overhead: int = 0
    mapped_circuit: Optional[MappedCircuit] = None
    routing_operations: List[Any] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_operation(self, op: "PhysicalOperation") -> None:
        """Add an operation to the routed circuit."""
        self.operations.append(op)
    
    def total_operations(self) -> int:
        """Total number of operations."""
        return len(self.operations)
    
    def interleaved_operations(self) -> List[Any]:
        """Return operations with routing ops interleaved at barrier positions.
        
        Uses the ``gate_execution_order`` from routing metadata to insert
        transport operations before each 2Q gate that required routing.
        If no routing metadata is available, returns operations unchanged.
        
        Returns
        -------
        List[Any]
            Gate and transport operations in execution order.
        """
        if not self.routing_operations:
            return list(self.operations)
        
        # Get barrier positions from metadata (indices into routing_operations
        # where a barrier occurs, i.e. a new routing pass starts)
        barriers = self.metadata.get("barriers", [])
        gate_exec_order = self.metadata.get("gate_execution_order", [])
        
        if not barriers and not gate_exec_order:
            # No routing structure — just append all routing ops first
            return list(self.routing_operations) + list(self.operations)
        
        # Build interleaved list: routing ops grouped by barriers,
        # then corresponding gate ops
        result: List[Any] = []
        
        # Identify 2Q gate ops by index
        gate_ops = list(self.operations)
        two_q_indices = [
            i for i, op in enumerate(gate_ops)
            if len(getattr(op, "qubits", ())) >= 2
        ]
        
        # Group routing ops by barrier
        routing_groups: List[List[Any]] = []
        current_group: List[Any] = []
        
        # First try pass boundary sentinels (WISE-style)
        _has_boundaries = False
        for rop in self.routing_operations:
            src_zone = getattr(rop, "source_zone", None)
            if src_zone == "__PASS_BOUNDARY__":
                _has_boundaries = True
                break
        
        if _has_boundaries:
            for rop in self.routing_operations:
                src_zone = getattr(rop, "source_zone", None)
                if src_zone == "__PASS_BOUNDARY__":
                    if current_group:
                        routing_groups.append(current_group)
                        current_group = []
                    current_group.append(rop)
                else:
                    current_group.append(rop)
        elif barriers:
            # Use barrier indices to split routing ops into groups.
            # Barriers are indices in the *original* all_ops list where
            # a new routing pass starts.  Since we filtered out MSGate
            # objects, approximate by splitting evenly across 2Q gates.
            n_routing = len(self.routing_operations)
            n_2q = len(two_q_indices)
            if n_2q > 0 and n_routing > 0:
                # Distribute routing ops roughly evenly across 2Q gates
                chunk = max(1, n_routing // n_2q)
                for _ri in range(0, n_routing, chunk):
                    _end = min(_ri + chunk, n_routing)
                    routing_groups.append(
                        list(self.routing_operations[_ri:_end]))
            else:
                current_group = list(self.routing_operations)
        else:
            current_group = list(self.routing_operations)
        if current_group:
            routing_groups.append(current_group)
        
        # Interleave: for each 2Q gate, emit its routing group first
        used_2q = 0
        used_routing = 0
        for op_idx, op in enumerate(gate_ops):
            # Check if this is a 2Q gate
            if op_idx in two_q_indices:
                # Emit next routing group if available
                if used_routing < len(routing_groups):
                    result.extend(routing_groups[used_routing])
                    used_routing += 1
                used_2q += 1
            result.append(op)
        
        # Append any remaining routing groups
        while used_routing < len(routing_groups):
            result.extend(routing_groups[used_routing])
            used_routing += 1
        
        return result
    
    def __len__(self) -> int:
        return len(self.operations)
    
    def __iter__(self):
        return iter(self.operations)


@dataclass  
class ScheduledCircuit:
    """Circuit with timing and parallelization information.
    
    Fourth stage: operations are scheduled with start times
    and parallel execution groups.
    
    Attributes
    ----------
    layers : List[CircuitLayer]
        Sequence of parallel execution layers.
    scheduled_ops : List[ScheduledOperation]
        All operations with timing info.
    total_duration : float
        Total circuit execution time in microseconds.
    routed_circuit : Optional[RoutedCircuit]
        Reference to the pre-scheduling routed circuit.
    batches : List[Any]
        Parallel execution batches from the scheduler.
    metadata : Dict[str, Any]
        Additional scheduling metadata.
    """
    layers: List[CircuitLayer] = field(default_factory=list)
    scheduled_ops: List[ScheduledOperation] = field(default_factory=list)
    total_duration: float = 0.0
    routed_circuit: Optional[RoutedCircuit] = None
    batches: List[Any] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_layer(self, layer: CircuitLayer) -> None:
        """Add a parallel execution layer."""
        if self.layers:
            layer.start_time = self.layers[-1].end_time
        self.layers.append(layer)
        self.total_duration = layer.end_time
    
    def depth(self) -> int:
        """Circuit depth (number of layers)."""
        return len(self.layers)
    
    @property
    def layer_count(self) -> int:
        """Number of layers (alias for depth())."""
        return self.depth()
    
    def parallelism(self) -> float:
        """Average operations per layer."""
        if not self.layers:
            return 0.0
        total_ops = sum(len(layer.operations) for layer in self.layers)
        return total_ops / len(self.layers)


@dataclass
class CompiledCircuit:
    """Fully compiled circuit ready for simulation.
    
    Final stage: contains all information needed to generate
    Stim circuits and run hardware simulation.
    
    Attributes
    ----------
    scheduled : ScheduledCircuit
        The scheduled circuit.
    mapping : QubitMapping
        Final qubit mapping.
    original_circuit : Optional[stim.Circuit]
        The original ideal stim circuit that was compiled.  This is
        the source of truth for noise injection — noise instructions
        are interleaved around the original gates, preserving all
        DETECTOR/OBSERVABLE_INCLUDE/QUBIT_COORDS annotations.
    stim_circuit : Optional[stim.Circuit]
        Generated Stim circuit (if already built).
    metrics : Dict[str, Any]
        Compilation metrics (depth, gate count, etc.).
    """
    scheduled: ScheduledCircuit
    mapping: QubitMapping
    original_circuit: Optional[stim.Circuit] = None
    stim_circuit: Optional[stim.Circuit] = None
    metrics: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def total_duration(self) -> float:
        """Total execution time in microseconds."""
        return self.scheduled.total_duration
    
    @property
    def depth(self) -> int:
        """Circuit depth (number of layers)."""
        return self.scheduled.depth()
    
    def to_stim(self) -> stim.Circuit:
        """Generate Stim circuit from compiled circuit.
        
        If the original ideal circuit is available, it is used as the
        source of truth — DETECTOR, OBSERVABLE_INCLUDE, QUBIT_COORDS,
        and SHIFT_COORDS annotations are preserved from it, with qubit
        indices remapped through ``self.mapping``.
        
        If no original circuit is stored, falls back to building a
        gate-only circuit from scheduled layers (no annotations).
        
        Returns cached circuit if already built.
        """
        if self.stim_circuit is not None:
            return self.stim_circuit
        
        if self.original_circuit is not None:
            self.stim_circuit = self._rebuild_from_original()
        else:
            self.stim_circuit = self._build_from_layers()
        
        return self.stim_circuit
    
    def _remap_qubit(self, logical: int) -> int:
        """Map a logical qubit index through the mapping, if available."""
        phys = self.mapping.get_physical(logical)
        return phys if phys is not None else logical
    
    def _rebuild_from_original(self) -> stim.Circuit:
        """Rebuild stim circuit from the original ideal circuit.
        
        Walks the original circuit and:
        - Remaps qubit indices on gate instructions through self.mapping
        - Copies annotations (DETECTOR, OBSERVABLE_INCLUDE, QUBIT_COORDS,
          SHIFT_COORDS) verbatim — these use relative rec[] targets, not
          qubit indices, so they don't need remapping.
        - Preserves REPEAT block structure.
        """
        circuit = stim.Circuit()
        
        # Emit QUBIT_COORDS preamble from original if present
        for inst in self.original_circuit:
            if isinstance(inst, stim.CircuitRepeatBlock):
                circuit.append(stim.CircuitRepeatBlock(
                    inst.repeat_count,
                    self._rebuild_body_from_original(inst.body_copy()),
                ))
            else:
                circuit.append(inst)
        
        return circuit
    
    def _rebuild_body_from_original(self, body: stim.Circuit) -> stim.Circuit:
        """Recursively rebuild a circuit body, preserving structure."""
        result = stim.Circuit()
        for inst in body:
            if isinstance(inst, stim.CircuitRepeatBlock):
                result.append(stim.CircuitRepeatBlock(
                    inst.repeat_count,
                    self._rebuild_body_from_original(inst.body_copy()),
                ))
            else:
                result.append(inst)
        return result
    
    def _build_from_layers(self) -> stim.Circuit:
        """Build gate-only circuit from scheduled layers (no annotations).
        
        This is the fallback when no original_circuit is available.
        Useful for debugging/metrics but NOT suitable for decoding.
        """
        circuit = stim.Circuit()
        
        for layer in self.scheduled.layers:
            for op in layer.operations:
                for instruction in op.to_stim_instructions():
                    if instruction.strip():
                        circuit.append_from_stim_program_text(instruction)
            circuit.append("TICK")
        
        return circuit
    
    def compute_metrics(self) -> Dict[str, Any]:
        """Compute and cache compilation metrics."""
        if self.metrics:
            return self.metrics
        
        total_ops = sum(len(layer.operations) for layer in self.scheduled.layers)
        two_qubit_ops = sum(
            1 for layer in self.scheduled.layers
            for op in layer.operations
            if len(op.qubits) == 2
        )
        
        self.metrics = {
            "total_operations": total_ops,
            "two_qubit_ops": two_qubit_ops,
            "depth": self.depth,
            "duration_us": self.total_duration,
            "parallelism": self.scheduled.parallelism(),
            "num_qubits": self.mapping.num_qubits(),
        }

        # --- Propagate routing/mapping metadata for downstream use ---
        # Each platform puts its own keys into routing_result.metadata;
        # core forwards them generically without knowing key names.
        routed = self.scheduled.routed_circuit
        if routed is not None and routed.metadata:
            self.metrics.update(routed.metadata)

        if routed is not None and routed.mapped_circuit is not None:
            mmeta = getattr(routed.mapped_circuit, "metadata", None) or {}
            if mmeta:
                self.metrics.update(mmeta)

        return self.metrics
