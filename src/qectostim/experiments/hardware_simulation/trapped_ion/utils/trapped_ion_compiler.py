"""
Trapped-ion hardware compiler.

Implements :class:`HardwareCompiler` for QCCD trapped-ion architectures.
The compilation pipeline:

1. **Decompose** — stim gates → native MS + rotation operations
2. **Map** — logical qubits → physical ions (cluster partitioning)
3. **Route** — ion shuttling via SAT-based WISE or heuristic routing
4. **Schedule** — parallel batches with timing
"""

from __future__ import annotations

import stim
from typing import (
    Sequence,
    List,
    Tuple,
    Optional,
    Dict,
    Any,
)

from qectostim.experiments.hardware_simulation.core.compiler import (
    HardwareCompiler,
)
from qectostim.experiments.hardware_simulation.core.pipeline import (
    NativeCircuit,
    MappedCircuit,
    RoutedCircuit,
    ScheduledCircuit,
    QubitMapping,
    CircuitLayer,
)

from .qccd_nodes import (
    Ion,
    QubitIon,
    SpectatorIon,
    QCCDWiseArch,
)
from .qccd_operations import Operation
from .qccd_operations_on_qubits import (
    QubitOperation,
    TwoQubitMSGate,
    OneQubitGate,
    XRotation,
    YRotation,
    Measurement,
    QubitReset,
)
from .gate_decomposition import decompose_stim_gate, DecomposedGate
from .qccd_arch import QCCDArch

from ..compiler.qccd_qubits_to_ions import (
    regularPartition,
    hillClimbOnArrangeClusters,
    arrangeClusters,
)
from ..compiler.qccd_parallelisation import (
    paralleliseOperations,
    paralleliseOperationsWithBarriers,
)
from ..compiler.qccd_ion_routing import ionRouting
from ..compiler.qccd_WISE_ion_route import ionRoutingWISEArch
from ..compiler.routing_config import (
    WISERoutingConfig,
    WISESolverParams,
    make_tqdm_progress_callback,
    make_nested_tqdm_progress_callback,
    make_triple_tqdm_progress_callback,
    make_single_tqdm_progress_callback,
    make_sat_only_tqdm_progress_callback,
    make_logging_progress_callback,
)


class TrappedIonCompiler(HardwareCompiler):
    """Compiler for QCCD trapped-ion hardware.

    Main entry point is :meth:`compile`, which runs:

    1. ``decompose_to_native`` — stim ops → MS + R{X,Y} + M + R
    2. ``map_qubits`` — cluster partitioning, architecture topology build
    3. ``route`` — ion shuttling (WISE SAT or heuristic)
    4. ``schedule`` — parallel batches with timing

    Parameters
    ----------
    architecture : QCCDArch
        Target QCCD architecture (may be a subclass).
    optimization_level : int
        Optimisation level (0-2).
    is_wise : bool
        If ``True``, uses WISE routing (``ionRoutingWISEArch``).
    wise_config : Optional[QCCDWiseArch]
        WISE configuration (required when ``is_wise=True``).
    data_qubit_idxs : Optional[Sequence[int]]
        Explicit data qubit indices.  When ``None``, data/ancilla
        classification uses QUBIT_COORDS parity.
    show_progress : bool
        If ``True`` (default), display a tqdm progress bar during
        WISE SAT routing.  The bar renders as a Jupyter widget in
        notebooks or a terminal bar in scripts.
    """

    def __init__(
        self,
        architecture: QCCDArch,
        optimization_level: int = 1,
        is_wise: bool = False,
        wise_config: Optional[QCCDWiseArch] = None,
        data_qubit_idxs: Optional[Sequence[int]] = None,
        show_progress: bool = True,
    ):
        self.is_wise = is_wise
        self.wise_config = wise_config
        self.data_qubit_idxs = data_qubit_idxs
        self.show_progress = show_progress

        # State populated during decompose_to_native
        self._ion_mapping: Dict[int, Tuple[Ion, Tuple[int, int]]] = {}
        self._measurement_ions: List[Ion] = []
        self._data_ions: List[Ion] = []
        self._instructions: List[QubitOperation] = []
        self._barriers: List[int] = []
        self._toMoveOps: List[List[TwoQubitMSGate]] = []
        self.routing_kwargs: Dict[str, Any] = {}

        super().__init__(architecture, optimization_level)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def ion_mapping(self) -> Dict[int, Tuple[Ion, Tuple[int, int]]]:
        return self._ion_mapping

    @property
    def measurement_ions(self) -> List[Ion]:
        return self._measurement_ions

    @property
    def data_ions(self) -> List[Ion]:
        return self._data_ions

    # ------------------------------------------------------------------
    # Abstract method implementations
    # ------------------------------------------------------------------

    def _setup_passes(self) -> None:
        """Passes are run manually via the pipeline methods."""
        pass

    def decompose_to_native(self, circuit: stim.Circuit) -> NativeCircuit:
        """Decompose stim circuit to native MS + rotation operations.

        This replaces ``QCCDCircuit.circuitString`` +
        ``QCCDCircuit._parseCircuitString``.

        The decomposition rules match those in the original:
        - H → (RY, RX)
        - CNOT/CX → (RY, RX, RX, MS, RY)
        - CZ → (RY, RX, RY, RX, MS, RY, RY, RX)
        - MR → (M, R)
        - Others → ``decompose_stim_gate`` fallback

        Returns
        -------
        NativeCircuit
            Contains the ``QubitOperation`` list in ``metadata["operations"]``,
            plus ``metadata["barriers"]``, ``metadata["toMoveOps"]``,
            ``metadata["ion_mapping"]``.
        """
        # --- 1. Flatten circuit to instruction strings ---
        instructions_raw = (
            circuit.flattened()
            .decomposed()
            .without_noise()
            .__str__()
            .splitlines()
        )

        # --- 1a. Build decomposed→sidebar-entry mapping ---
        # The sidebar is built from ``str(circuit)`` (original, un-
        # flattened), while the compiler works from ``circuit.flattened()
        # .decomposed()``.  We need to connect each decomposed gate line
        # back to its sidebar entry index.
        #
        # Strategy:
        #   1. Walk the *original* circuit lines to build a gate-text →
        #      sidebar-index lookup (preserving order for duplicates).
        #   2. Walk the *flattened* circuit gate lines.  Each flattened
        #      gate must correspond to an original gate (flattening only
        #      expands REPEATs; simple circuits are unchanged).  Look up
        #      its sidebar index.
        #   3. Decompose each flattened gate to primitives to count how
        #      many decomposed lines it produces, assigning each the
        #      same sidebar index.

        _skip_prefixes = ("QUBIT_COORDS", "DETECTOR", "TICK", "OBSERVABLE",
                          "SHIFT_COORDS", "REPEAT", "}", "MPP")

        # Step 1: original circuit → sidebar entry indices for gate lines
        _orig_lines = str(circuit).splitlines()
        _orig_gate_sidebar: List[Tuple[str, int]] = []   # (gate_text, sidebar_idx)
        _sidebar_idx = 0
        for line in _orig_lines:
            stripped = line.strip()
            if not stripped:
                continue
            is_gate = not any(stripped.upper().startswith(p) for p in _skip_prefixes)
            if is_gate:
                _orig_gate_sidebar.append((stripped, _sidebar_idx))
            _sidebar_idx += 1

        # Step 2: flattened circuit gate lines → sidebar index
        # For simple (non-REPEAT) circuits, flattened == original gate lines.
        # For REPEAT circuits, we match gate text sequentially.
        _flat_lines = str(circuit.flattened()).splitlines()
        _flat_gate_lines = [l.strip() for l in _flat_lines
                            if l.strip() and not any(
                                l.strip().upper().startswith(p)
                                for p in _skip_prefixes)]

        _flat_gate_to_sidebar: List[int] = []
        _orig_cursor = 0
        for fg in _flat_gate_lines:
            # Find the next matching original gate (handles REPEAT expansion)
            matched = False
            # For non-repeat circuits, gates match 1:1 in order
            if _orig_cursor < len(_orig_gate_sidebar):
                og_text, og_sb = _orig_gate_sidebar[_orig_cursor]
                if og_text == fg:
                    _flat_gate_to_sidebar.append(og_sb)
                    _orig_cursor += 1
                    matched = True
            if not matched:
                # REPEAT expansion: reuse the last block's sidebar indices
                # by cycling through them
                _flat_gate_to_sidebar.append(-1)

        # Step 3: decompose each flattened gate → count primitives
        _decomp_to_sidebar: Dict[int, int] = {}
        _dcursor = 0
        for _fgi, _fg in enumerate(_flat_gate_lines):
            try:
                _mini = stim.Circuit(_fg).decomposed().__str__().splitlines()
                _mini_gates = [l for l in _mini
                               if l.strip() and not any(
                                   l.strip().upper().startswith(p)
                                   for p in _skip_prefixes)]
            except Exception:
                _mini_gates = [_fg]
            _sb_idx = _flat_gate_to_sidebar[_fgi] if _fgi < len(_flat_gate_to_sidebar) else -1
            for _ in _mini_gates:
                _decomp_to_sidebar[_dcursor] = _sb_idx
                _dcursor += 1

        newInstructions: List[str] = []
        new_instr_origin: List[int] = []   # parallel to newInstructions
        new_instr_epoch: List[int] = []    # TICK epoch for each instruction
        toMoves: List[List[Tuple[str, str]]] = []
        _decomp_gate_idx = 0
        _tick_epoch = 0                    # incremented at each TICK
        for instr in instructions_raw:
            qubits = instr.rsplit(" ")[1:]
            if instr.startswith("TICK"):
                _tick_epoch += 1
                continue
            if instr.startswith("DETECTOR") or instr.startswith("OBSERVABLE"):
                continue

            # Determine the original-gate index for this decomposed line
            _is_gate_line = not instr.startswith("QUBIT_COORDS")
            if _is_gate_line:
                _origin = _decomp_to_sidebar.get(_decomp_gate_idx, -1)
                _decomp_gate_idx += 1
            else:
                _origin = -1

            if instr[0] in ("R", "H", "M"):
                for qubit in qubits:
                    newInstructions.append(f"{instr[0]} {qubit}")
                    new_instr_origin.append(_origin)
                    new_instr_epoch.append(_tick_epoch)
            elif any(instr.startswith(s) for s in stim.gate_data("cnot").aliases):
                toMove: List[Tuple[str, str]] = []
                for j in range(int(len(qubits) / 2)):
                    newInstructions.append(f"CNOT {qubits[2*j]} {qubits[2*j+1]}")
                    new_instr_origin.append(_origin)
                    new_instr_epoch.append(_tick_epoch)
                    toMove.append((qubits[2*j], qubits[2*j+1]))
                toMoves.append(toMove)
                newInstructions.append("BARRIER")
                new_instr_origin.append(-1)
                new_instr_epoch.append(_tick_epoch)
            elif any(instr.startswith(s) for s in stim.gate_data("cz").aliases):
                toMove = []
                for j in range(int(len(qubits) / 2)):
                    newInstructions.append(f"CZ {qubits[2*j]} {qubits[2*j+1]}")
                    new_instr_origin.append(_origin)
                    new_instr_epoch.append(_tick_epoch)
                    toMove.append((qubits[2*j], qubits[2*j+1]))
                toMoves.append(toMove)
                newInstructions.append("BARRIER")
                new_instr_origin.append(-1)
                new_instr_epoch.append(_tick_epoch)
            else:
                newInstructions.append(instr)
                new_instr_origin.append(_origin)
                new_instr_epoch.append(_tick_epoch)

        # --- 2. Parse QUBIT_COORDS → create ions ---
        self._measurement_ions = []
        self._ion_mapping = {}
        self._data_ions = []

        j = 0
        for j, instr in enumerate(newInstructions):
            if not instr.startswith("QUBIT_COORDS"):
                break
            coords_str = instr.removeprefix("QUBIT_COORDS(").split(")")[0].split(",")
            coords = tuple(int(float(c.strip())) for c in coords_str)
            idx = int(instr.split(" ")[-1])
            if self.data_qubit_idxs is not None or (coords[0] % 2) == 0:
                ion = QubitIon("#e8927c", label="M")
                ion.set(ion.idx, *coords)
                self._ion_mapping[idx] = ion, coords
                self._measurement_ions.append(ion)
            else:
                ion = QubitIon("#7fc7af", label="D")
                ion.set(ion.idx, *coords)
                self._ion_mapping[idx] = ion, coords
                self._data_ions.append(ion)

        remaining = newInstructions[j:]
        remaining_origin = new_instr_origin[j:]
        remaining_epoch = new_instr_epoch[j:]

        # --- 3. Parse instructions → QubitOperation list ---
        #     Each QubitOperation is tagged with ``_stim_origin`` = int,
        #     the index of the *original* (pre-decomposition) stim gate
        #     instruction it traces back to.  This directly corresponds
        #     to the sidebar entry order from ``parse_stim_for_sidebar``.
        #     Each is also tagged with ``_tick_epoch`` = int, the TICK
        #     epoch from the decomposed stim circuit for happens-before
        #     enforcement.
        operations: List[QubitOperation] = []
        barriers: List[int] = []
        dataQubits: List[Ion] = []
        toMoveOps: List[List[TwoQubitMSGate]] = [
            [] for _ in range(len(toMoves))
        ]
        toMoveIdx = 0

        def _tag(op: QubitOperation, origin: int, epoch: int) -> QubitOperation:
            """Attach stim provenance and tick epoch to a native operation."""
            op._stim_origin = origin  # type: ignore[attr-defined]
            op._tick_epoch = epoch    # type: ignore[attr-defined]
            return op

        for _ri, instr in enumerate(remaining):
            _origin = remaining_origin[_ri] if _ri < len(remaining_origin) else -1
            _epoch = remaining_epoch[_ri] if _ri < len(remaining_epoch) else -1
            if instr.startswith("BARRIER"):
                barriers.append(len(operations))
                continue
            if instr.startswith((
                "TICK", "DETECTOR", "OBSERVABLE_INCLUDE",
                "SHIFT_COORDS", "QUBIT_COORDS", "REPEAT", "}", "MPP",
            )):
                continue
            parts = instr.split()
            if not parts:
                continue
            gate_name = parts[0]
            qubit_parts = [p for p in parts[1:] if not p.startswith("(")]
            if not qubit_parts:
                continue
            try:
                idx = int(qubit_parts[0])
                ion = self._ion_mapping[idx][0]
            except (ValueError, KeyError):
                continue

            if gate_name in ("M", "MZ"):
                operations.append(_tag(Measurement.qubitOperation(ion), _origin, _epoch))
            elif gate_name == "H":
                operations.extend([
                    _tag(YRotation.qubitOperation(ion), _origin, _epoch),
                    _tag(XRotation.qubitOperation(ion), _origin, _epoch),
                ])
            elif gate_name == "R":
                operations.append(_tag(QubitReset.qubitOperation(ion), _origin, _epoch))
                if self.data_qubit_idxs is None:
                    dataQubits.clear()
            elif gate_name in ("CNOT", "CX", "ZCX"):
                idx2 = int(qubit_parts[1])
                ion2 = self._ion_mapping[idx2][0]
                operations.extend([
                    _tag(YRotation.qubitOperation(ion), _origin, _epoch),
                    _tag(XRotation.qubitOperation(ion), _origin, _epoch),
                    _tag(XRotation.qubitOperation(ion2), _origin, _epoch),
                    _tag(TwoQubitMSGate.qubitOperation(ion, ion2), _origin, _epoch),
                    _tag(YRotation.qubitOperation(ion), _origin, _epoch),
                ])
                toMoveOps[toMoveIdx].append(operations[-2])
                if len(toMoveOps[toMoveIdx]) == len(toMoves[toMoveIdx]):
                    toMoveIdx += 1
            elif gate_name in ("CZ", "ZCZ"):
                idx2 = int(qubit_parts[1])
                ion2 = self._ion_mapping[idx2][0]
                operations.extend([
                    _tag(YRotation.qubitOperation(ion), _origin, _epoch),
                    _tag(XRotation.qubitOperation(ion), _origin, _epoch),
                    _tag(YRotation.qubitOperation(ion2), _origin, _epoch),
                    _tag(XRotation.qubitOperation(ion2), _origin, _epoch),
                    _tag(TwoQubitMSGate.qubitOperation(ion, ion2), _origin, _epoch),
                    _tag(YRotation.qubitOperation(ion), _origin, _epoch),
                    _tag(YRotation.qubitOperation(ion2), _origin, _epoch),
                    _tag(XRotation.qubitOperation(ion2), _origin, _epoch),
                ])
                toMoveOps[toMoveIdx].append(operations[-4])
                if len(toMoveOps[toMoveIdx]) == len(toMoves[toMoveIdx]):
                    toMoveIdx += 1
            elif gate_name == "MR":
                operations.append(_tag(Measurement.qubitOperation(ion), _origin, _epoch))
                operations.append(_tag(QubitReset.qubitOperation(ion), _origin, _epoch))
            else:
                # Fallback: gate decomposition module
                all_qubit_indices = tuple(int(p) for p in qubit_parts)
                try:
                    decomposed = decompose_stim_gate(gate_name, all_qubit_indices)
                    for dg in decomposed:
                        if dg.name == "MS" and len(dg.qubits) == 2:
                            q0_ion = self._ion_mapping[dg.qubits[0]][0]
                            q1_ion = self._ion_mapping[dg.qubits[1]][0]
                            operations.append(_tag(
                                TwoQubitMSGate.qubitOperation(q0_ion, q1_ion),
                                _origin, _epoch,
                            ))
                            toMoveOps[toMoveIdx].append(operations[-1])
                            if len(toMoveOps[toMoveIdx]) == len(toMoves[toMoveIdx]):
                                toMoveIdx += 1
                        elif dg.name in ("RX", "RY"):
                            dg_ion = self._ion_mapping[dg.qubits[0]][0]
                            if dg.name == "RX":
                                operations.append(_tag(
                                    XRotation.qubitOperation(dg_ion),
                                    _origin, _epoch,
                                ))
                            else:
                                operations.append(_tag(
                                    YRotation.qubitOperation(dg_ion),
                                    _origin, _epoch,
                                ))
                        elif dg.name == "RZ":
                            dg_ion = self._ion_mapping[dg.qubits[0]][0]
                            operations.append(_tag(
                                YRotation.qubitOperation(dg_ion),
                                _origin, _epoch,
                            ))
                        elif dg.name == "M":
                            dg_ion = self._ion_mapping[dg.qubits[0]][0]
                            operations.append(_tag(
                                Measurement.qubitOperation(dg_ion),
                                _origin, _epoch,
                            ))
                        elif dg.name in ("MX", "MY"):
                            dg_ion = self._ion_mapping[dg.qubits[0]][0]
                            operations.append(_tag(
                                Measurement.qubitOperation(dg_ion),
                                _origin, _epoch,
                            ))
                        elif dg.name == "R":
                            dg_ion = self._ion_mapping[dg.qubits[0]][0]
                            operations.append(_tag(
                                QubitReset.qubitOperation(dg_ion),
                                _origin, _epoch,
                            ))
                except (ValueError, KeyError):
                    pass

        if self.data_qubit_idxs is not None:
            dataQubits = [self._ion_mapping[j][0] for j in self.data_qubit_idxs]

        for d in dataQubits:
            d._color = "#7fc7af"
            d._label = "D"
            self._data_ions.append(d)
            if d in self._measurement_ions:
                self._measurement_ions.remove(d)

        self._instructions = operations
        self._barriers = barriers
        self._toMoveOps = toMoveOps

        return NativeCircuit(
            operations=operations,
            num_qubits=len(self._ion_mapping),
            metadata={
                "barriers": barriers,
                "toMoveOps": toMoveOps,
                "ion_mapping": self._ion_mapping,
                "measurement_ions": self._measurement_ions,
                "data_ions": self._data_ions,
                "num_tick_epochs": _tick_epoch + 1,
            },
            stim_source=circuit,
        )

    def map_qubits(self, circuit: NativeCircuit) -> MappedCircuit:
        """Map logical qubits to physical ions via cluster partitioning.

        Triggers architecture topology building.

        Returns
        -------
        MappedCircuit
            Contains the qubit mapping.
        """
        arch = self.architecture
        ion_mapping = circuit.metadata.get("ion_mapping", self._ion_mapping)
        measurement_ions = circuit.metadata.get(
            "measurement_ions", self._measurement_ions
        )
        data_ions = circuit.metadata.get("data_ions", self._data_ions)

        # Build architecture topology if method exists
        if hasattr(arch, "build_topology"):
            arch.build_topology(measurement_ions, data_ions, ion_mapping)

        # Build qubit mapping
        mapping = QubitMapping()
        for stim_idx, (ion, _coords) in ion_mapping.items():
            if isinstance(ion, SpectatorIon):
                continue
            mapping.add_mapping(stim_idx, ion.idx)

        return MappedCircuit(
            native_circuit=circuit,
            mapping=mapping,
            metadata={"ion_mapping": ion_mapping},
        )

    def route(self, circuit: MappedCircuit) -> RoutedCircuit:
        """Route ions via shuttling operations.

        Uses WISE routing or heuristic routing depending on
        ``self.is_wise``.

        Returns
        -------
        RoutedCircuit
            With routing operations inserted.
        """
        arch = self.architecture
        arch.refreshGraph()

        instructions = circuit.native_circuit.metadata.get(
            "operations",
            circuit.native_circuit.operations,
        )
        barriers = circuit.native_circuit.metadata.get("barriers", [])

        reconfig_time = 0.0
        _progress_close = None            # might be set below for tqdm cleanup
        if self.is_wise and self.wise_config is not None:
            kwargs = dict(self.routing_kwargs)
            if "routing_config" not in kwargs or kwargs.get("routing_config") is None:
                # No routing_config provided — use production defaults
                # (all CPU cores, base_pmax_in=1, tqdm progress).
                kwargs["routing_config"] = WISERoutingConfig.default(
                    show_progress=self.show_progress,
                )
                rc = kwargs["routing_config"]
                if rc.progress_callback is not None:
                    # Stash the close function so we can clean up later
                    _progress_close = getattr(rc, '_progress_close', None)
            else:
                # routing_config was pre-provided — inject tqdm if
                # show_progress is on and no callback was set
                rc = kwargs["routing_config"]
                if self.show_progress and rc is not None and rc.progress_callback is None:
                    _progress_cb, _progress_close = make_triple_tqdm_progress_callback(
                        round_desc="MS Rounds",
                        patch_desc="Patches",
                        sat_desc="SAT Configs",
                    )
                    rc.progress_callback = _progress_cb
            # Auto-populate top-level ionRoutingWISEArch kwargs from
            # routing_config so callers only need to set values once.
            rc = kwargs.get("routing_config")
            if rc is not None:
                if "lookahead" not in kwargs:
                    kwargs["lookahead"] = rc.lookahead
                if "subgridsize" not in kwargs:
                    kwargs["subgridsize"] = rc.subgridsize
                if "base_pmax_in" not in kwargs and rc.base_pmax_in is not None:
                    kwargs["base_pmax_in"] = rc.base_pmax_in
                if "max_inner_workers" not in kwargs:
                    kwargs["max_inner_workers"] = rc.sat_workers

            # Pass pre-grouped toMoveOps so the SAT router uses the same
            # CX groupings produced by decompose_to_native.
            if "toMoveOps" not in kwargs:
                _toMoveOps = circuit.native_circuit.metadata.get("toMoveOps")
                if _toMoveOps is not None:
                    kwargs["toMoveOps"] = _toMoveOps
            allOps, route_barriers, reconfig_time = ionRoutingWISEArch(
                arch, self.wise_config, instructions, **kwargs
            )
            # Close tqdm bar if we created one
            if _progress_close is not None:
                _progress_close()
        else:
            trap_capacity = getattr(arch, "trap_capacity", 2)
            allOps, route_barriers = ionRouting(
                arch, instructions, trap_capacity
            )

        routing_only = [
            op for op in allOps if not isinstance(op, QubitOperation)
        ]

        return RoutedCircuit(
            operations=allOps,
            final_mapping=circuit.mapping,
            routing_overhead=len(routing_only),
            mapped_circuit=circuit,
            routing_operations=routing_only,
            metadata={
                "barriers": route_barriers if route_barriers else barriers,
                "all_operations": allOps,
                "is_wise": self.is_wise,
                "reconfig_time": reconfig_time,
            },
        )

    def schedule(self, circuit: RoutedCircuit) -> ScheduledCircuit:
        """Schedule operations into parallel batches.

        Returns
        -------
        ScheduledCircuit
            With timing information.
        """
        barriers = circuit.metadata.get("barriers", [])
        is_wise = circuit.metadata.get("is_wise", False)
        allOps = circuit.operations

        if barriers:
            parallelOpsMap = paralleliseOperationsWithBarriers(
                allOps, barriers, isWiseArch=is_wise
            )
        else:
            parallelOpsMap = paralleliseOperations(
                allOps, isWISEArch=is_wise
            )

        # Convert to ScheduledCircuit format
        layers: List[CircuitLayer] = []
        for _time_step, par_op in sorted(parallelOpsMap.items()):
            par_op.calculateOperationTime()
            par_op.calculateFidelity()
            layer = CircuitLayer(
                operations=par_op.operations,
                start_time=float(_time_step),
            )
            layers.append(layer)

        total_dur = max(parallelOpsMap.keys()) if parallelOpsMap else 0.0

        return ScheduledCircuit(
            layers=layers,
            total_duration=float(total_dur),
            routed_circuit=circuit,
            batches=list(parallelOpsMap.values()),
            metadata={
                "parallel_ops_map": parallelOpsMap,
                "all_operations": list(allOps),
            },
        )
