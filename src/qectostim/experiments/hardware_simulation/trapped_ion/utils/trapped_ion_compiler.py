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

import logging
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
    reorder_rotations_for_batching,
)
from ..compiler.qccd_ion_routing import ionRouting
from ..compiler.qccd_WISE_ion_route import ionRoutingWISEArch, ionRoutingGadgetArch
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

logger = logging.getLogger(__name__)


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
        # --- 1. Build decomposed→sidebar-entry mapping ---
        # The sidebar is built from ``str(circuit)`` (original, un-
        # flattened), while the compiler works from decomposed gate
        # lines.  We need to connect each decomposed gate line back
        # to its sidebar entry index.
        #
        # Strategy:
        #   1. Walk the *original* circuit lines to build a gate-text →
        #      sidebar-index lookup (preserving order for duplicates).
        #   2. Walk the *flattened* circuit gate lines.  Each flattened
        #      gate must correspond to an original gate (flattening only
        #      expands REPEATs; simple circuits are unchanged).  Look up
        #      its sidebar index.
        #   3. Decompose each flattened gate *individually* and build
        #      ``instructions_raw`` from the per-gate results, keeping
        #      non-gate lines (TICK, DETECTOR, QUBIT_COORDS, etc.)
        #      from the flattened circuit.  This avoids the merging
        #      of adjacent same-type gates that ``stim.Circuit.
        #      decomposed()`` performs on the whole circuit (e.g.
        #      ``R 6`` + ``R 11 12`` → ``R 6 11 12``), which would
        #      desynchronise the per-line sidebar mapping.

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
        _flat_circuit = circuit.flattened().without_noise()
        _flat_lines = str(_flat_circuit).splitlines()
        _flat_gate_lines = [l.strip() for l in _flat_lines
                            if l.strip() and not any(
                                l.strip().upper().startswith(p)
                                for p in _skip_prefixes)]

        # Build reverse lookup: gate_text → list of sidebar indices
        # (for REPEAT body fallback when forward matching exhausts)
        _text_to_sidebar: Dict[str, List[int]] = {}
        for _txt, _sb in _orig_gate_sidebar:
            _text_to_sidebar.setdefault(_txt, []).append(_sb)

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
                # REPEAT expansion: reuse the REPEAT body's sidebar
                # indices by looking up the gate text.  Prefer the last
                # occurrence (the REPEAT body entry) over earlier entries
                # so that all repeated iterations highlight the body.
                candidates = _text_to_sidebar.get(fg, [])
                if candidates:
                    _flat_gate_to_sidebar.append(candidates[-1])
                else:
                    _flat_gate_to_sidebar.append(-1)

        # Step 3: Build instructions_raw by decomposing each flattened
        # gate individually, interleaved with non-gate lines from the
        # flattened circuit.  This also builds _decomp_to_sidebar.
        #
        # Using whole-circuit decomposed() would merge adjacent same-
        # type gate lines (e.g. ``R 6`` from ``R 6`` + ``R 11 12``
        # from ``RX 11 12`` → ``R 6 11 12``), throwing off the index
        # alignment.  Per-gate decomposition avoids this.
        instructions_raw: List[str] = []
        _decomp_to_sidebar: Dict[int, int] = {}
        _flat_gate_cursor = 0
        _decomp_gate_idx = 0

        for line in _flat_lines:
            stripped = line.strip()
            if not stripped:
                continue
            upper = stripped.upper()
            is_gate = not any(upper.startswith(p) for p in _skip_prefixes)
            if not is_gate:
                # Pass through non-gate lines (TICK, DETECTOR, etc.)
                instructions_raw.append(stripped)
                continue
            # Decompose this single gate individually
            _sb_idx = (_flat_gate_to_sidebar[_flat_gate_cursor]
                       if _flat_gate_cursor < len(_flat_gate_to_sidebar)
                       else -1)
            _flat_gate_cursor += 1
            try:
                _mini = stim.Circuit(stripped).decomposed().__str__().splitlines()
                _mini_gates = [l.strip() for l in _mini
                               if l.strip() and not any(
                                   l.strip().upper().startswith(p)
                                   for p in _skip_prefixes)]
            except (ValueError, RuntimeError) as _decomp_exc:
                logger.warning(
                    "stim decomposed() failed for line %r: %s — using raw line",
                    stripped, _decomp_exc,
                )
                _mini_gates = [stripped]
            for mg in _mini_gates:
                instructions_raw.append(mg)
                _decomp_to_sidebar[_decomp_gate_idx] = _sb_idx
                _decomp_gate_idx += 1

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

        # C4 fix: Track expected skips per toMoveOps bucket so that
        # toMoveIdx advances correctly when CX pairs are skipped.
        _toMove_skips: List[int] = [0] * len(toMoves)

        def _advance_toMoveIdx_if_full():
            """Advance toMoveIdx when bucket is full (including skips)."""
            nonlocal toMoveIdx
            while (
                toMoveIdx < len(toMoves)
                and (len(toMoveOps[toMoveIdx]) + _toMove_skips[toMoveIdx])
                    >= len(toMoves[toMoveIdx])
            ):
                toMoveIdx += 1

        def _record_cx_skip():
            """Record that a CX pair was skipped in the current bucket."""
            if toMoveIdx < len(toMoves):
                _toMove_skips[toMoveIdx] += 1
                _advance_toMoveIdx_if_full()

        def _tag(op: QubitOperation, origin: int, epoch: int) -> QubitOperation:
            """Attach stim provenance and tick epoch to a native operation."""
            op._stim_origin = origin  # type: ignore[attr-defined]
            op._tick_epoch = epoch    # type: ignore[attr-defined]
            return op

        # ── Helper: resolve qubit index → Ion ───────────────
        def _ion_for(qi_str):
            return self._ion_mapping[int(qi_str)][0]

        # H9: Track skipped ops for diagnostics
        _skipped_ops = 0

        def _log_skip(gate: str, qubits):
            """Record a skipped gate for H9 diagnostics."""
            nonlocal _skipped_ops
            _skipped_ops += 1
            logger.debug(
                "decompose_to_native: skipped %s gate (qubit(s) %s "
                "not in ion_mapping) — total skips: %d",
                gate, qubits, _skipped_ops,
            )

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

            # ── Single-qubit gates: iterate ALL qubit targets ───
            if gate_name in ("M", "MZ"):
                for qi in qubit_parts:
                    try:
                        ion = _ion_for(qi)
                    except (ValueError, KeyError):
                        _log_skip(gate_name, qi)
                        continue
                    operations.append(_tag(Measurement.qubitOperation(ion), _origin, _epoch))

            elif gate_name == "H":
                for qi in qubit_parts:
                    try:
                        ion = _ion_for(qi)
                    except (ValueError, KeyError):
                        _log_skip(gate_name, qi)
                        continue
                    operations.extend([
                        _tag(YRotation.qubitOperation(ion), _origin, _epoch),
                        _tag(XRotation.qubitOperation(ion), _origin, _epoch),
                    ])

            elif gate_name == "R":
                for qi in qubit_parts:
                    try:
                        ion = _ion_for(qi)
                    except (ValueError, KeyError):
                        _log_skip(gate_name, qi)
                        continue
                    operations.append(_tag(QubitReset.qubitOperation(ion), _origin, _epoch))
                # M6: Removed dead `dataQubits.clear()`

            elif gate_name == "MR":
                for qi in qubit_parts:
                    try:
                        ion = _ion_for(qi)
                    except (ValueError, KeyError):
                        _log_skip(gate_name, qi)
                        continue
                    operations.append(_tag(Measurement.qubitOperation(ion), _origin, _epoch))
                    operations.append(_tag(QubitReset.qubitOperation(ion), _origin, _epoch))

            # ── Direct RX / MX / MRX handling (Fix E2) ────────
            elif gate_name == "RX":
                # X-basis reset: R → RY → RX  (prepare |+⟩)
                for qi in qubit_parts:
                    try:
                        ion = _ion_for(qi)
                    except (ValueError, KeyError):
                        _log_skip(gate_name, qi)
                        continue
                    operations.append(_tag(QubitReset.qubitOperation(ion), _origin, _epoch))
                    operations.append(_tag(YRotation.qubitOperation(ion), _origin, _epoch))
                    operations.append(_tag(XRotation.qubitOperation(ion), _origin, _epoch))

            elif gate_name in ("MX", "MY"):
                # Basis-change then measure
                for qi in qubit_parts:
                    try:
                        ion = _ion_for(qi)
                    except (ValueError, KeyError):
                        _log_skip(gate_name, qi)
                        continue
                    if gate_name == "MX":
                        operations.append(_tag(XRotation.qubitOperation(ion), _origin, _epoch))
                        operations.append(_tag(YRotation.qubitOperation(ion), _origin, _epoch))
                    else:  # MY
                        operations.append(_tag(YRotation.qubitOperation(ion), _origin, _epoch))
                    operations.append(_tag(Measurement.qubitOperation(ion), _origin, _epoch))

            elif gate_name == "MRX":
                # MX then RX: (XRot → YRot → M) + (R → YRot → XRot)
                for qi in qubit_parts:
                    try:
                        ion = _ion_for(qi)
                    except (ValueError, KeyError):
                        _log_skip(gate_name, qi)
                        continue
                    # MX part
                    operations.append(_tag(XRotation.qubitOperation(ion), _origin, _epoch))
                    operations.append(_tag(YRotation.qubitOperation(ion), _origin, _epoch))
                    operations.append(_tag(Measurement.qubitOperation(ion), _origin, _epoch))
                    # RX part
                    operations.append(_tag(QubitReset.qubitOperation(ion), _origin, _epoch))
                    operations.append(_tag(YRotation.qubitOperation(ion), _origin, _epoch))
                    operations.append(_tag(XRotation.qubitOperation(ion), _origin, _epoch))

            # ── Two-qubit gates: iterate over pairs ────────────
            elif gate_name in ("CNOT", "CX", "ZCX"):
                for pair_start in range(0, len(qubit_parts) - 1, 2):
                    try:
                        ion = _ion_for(qubit_parts[pair_start])
                        ion2 = _ion_for(qubit_parts[pair_start + 1])
                    except (ValueError, KeyError):
                        _log_skip(gate_name, qubit_parts[pair_start:pair_start+2])
                        _record_cx_skip()
                        continue
                    operations.extend([
                        _tag(YRotation.qubitOperation(ion), _origin, _epoch),
                        _tag(XRotation.qubitOperation(ion), _origin, _epoch),
                        _tag(XRotation.qubitOperation(ion2), _origin, _epoch),
                        _tag(TwoQubitMSGate.qubitOperation(ion, ion2), _origin, _epoch),
                        _tag(YRotation.qubitOperation(ion), _origin, _epoch),
                    ])
                    toMoveOps[toMoveIdx].append(operations[-2])
                    _advance_toMoveIdx_if_full()

            elif gate_name in ("CZ", "ZCZ"):
                for pair_start in range(0, len(qubit_parts) - 1, 2):
                    try:
                        ion = _ion_for(qubit_parts[pair_start])
                        ion2 = _ion_for(qubit_parts[pair_start + 1])
                    except (ValueError, KeyError):
                        _log_skip(gate_name, qubit_parts[pair_start:pair_start+2])
                        _record_cx_skip()
                        continue
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
                    _advance_toMoveIdx_if_full()
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
                            _advance_toMoveIdx_if_full()
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
                    _log_skip(gate_name, all_qubit_indices)

        if _skipped_ops > 0:
            logger.info(
                "decompose_to_native: total skipped ops: %d",
                _skipped_ops,
            )

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

        When ``qec_metadata`` with ≥ 2 block allocations is present in
        ``circuit.metadata``, automatically switches to **per-block
        topology building** (``build_topology_per_block``) so that each
        code block is placed in its own disjoint sub-grid region.

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

        # ── Detect multi-block gadget experiments ────────────────────
        qec_metadata = getattr(circuit, 'qec_metadata', None)
        if qec_metadata is None:
            qec_metadata = circuit.metadata.get("qec_metadata")
        qubit_allocation = circuit.metadata.get("qubit_allocation")

        _use_per_block = (
            qec_metadata is not None
            and qubit_allocation is not None
            and hasattr(qec_metadata, 'block_allocations')
            and len(qec_metadata.block_allocations) >= 2
            and hasattr(arch, 'build_topology_per_block')
        )

        if _use_per_block:
            return self._map_qubits_per_block(
                circuit, arch, ion_mapping, qec_metadata, qubit_allocation,
            )

        # ── Standard global topology ────────────────────────────────
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

    # ------------------------------------------------------------------
    # Per-block topology for multi-block gadget experiments
    # ------------------------------------------------------------------

    def _map_qubits_per_block(
        self,
        circuit: NativeCircuit,
        arch,
        ion_mapping: dict,
        qec_metadata,
        qubit_allocation,
    ) -> MappedCircuit:
        """Per-block topology building for multi-block gadgets.

        Mirrors the logic previously in ``compile_gadget_for_animation``:
        separates ions by block, assigns bridge ancillas, then calls
        ``arch.build_topology_per_block``.
        """
        from .gadget_routing import partition_grid_for_blocks

        k = getattr(self.wise_config, 'k', 2) if self.wise_config else 2

        sub_grids = partition_grid_for_blocks(
            qec_metadata, qubit_allocation, k=k,
        )

        block_names = [ba.block_name for ba in qec_metadata.block_allocations]

        # ── Bridge ancilla → block assignment ────────────────────────
        _bridge_block_map: Dict[int, str] = {}
        _explicit_bridge_map: Dict[int, list] = getattr(
            qubit_allocation, 'bridge_connected_blocks', {}
        ) or {}
        if hasattr(qubit_allocation, "bridge_ancillas"):
            for gi, _coord, purpose in qubit_allocation.bridge_ancillas:
                _assigned: Optional[str] = None
                if gi in _explicit_bridge_map and _explicit_bridge_map[gi]:
                    for _cb in _explicit_bridge_map[gi]:
                        if _cb in block_names:
                            _assigned = _cb
                            break
                if _assigned is None:
                    # Heuristic: assign bridge ancillas to blocks by
                    # merge type.  Use first/second block by index —
                    # works regardless of actual naming convention.
                    if "zz_merge" in purpose and len(block_names) >= 1:
                        _assigned = block_names[0]
                    elif "xx_merge" in purpose and len(block_names) >= 2:
                        _assigned = block_names[1]
                if _assigned is None and block_names:
                    _assigned = block_names[0]
                if _assigned is not None:
                    _bridge_block_map[gi] = _assigned

        # ── Separate ions by block ───────────────────────────────────
        measurement_ions_per_block: Dict[str, list] = {}
        data_ions_per_block: Dict[str, list] = {}
        for ba in qec_metadata.block_allocations:
            m_ions_block, d_ions_block = [], []
            all_qubits = (
                set(ba.data_qubits)
                | set(ba.x_ancilla_qubits)
                | set(ba.z_ancilla_qubits)
            )
            for gi, assigned_bn in _bridge_block_map.items():
                if assigned_bn == ba.block_name:
                    all_qubits.add(gi)
            for q in all_qubits:
                if q in ion_mapping:
                    ion, _ = ion_mapping[q]
                    if ion._label == "M":
                        m_ions_block.append(ion)
                    else:
                        d_ions_block.append(ion)
            measurement_ions_per_block[ba.block_name] = m_ions_block
            data_ions_per_block[ba.block_name] = d_ions_block

        # ── Build per-block topology ─────────────────────────────────
        arch.build_topology_per_block(
            sub_grids, measurement_ions_per_block,
            data_ions_per_block, ion_mapping,
        )

        # ── Update sub_grids.ion_indices AND qubit_to_ion with
        #    reassigned Ion.idx (Fix A) ─────────────────────────────
        for ba in qec_metadata.block_allocations:
            new_ion_indices = []
            new_q2i: Dict[int, int] = {}  # Fix A: physical mapping
            all_qubits = (
                set(ba.data_qubits)
                | set(ba.x_ancilla_qubits)
                | set(ba.z_ancilla_qubits)
            )
            for gi, assigned_bn in _bridge_block_map.items():
                if assigned_bn == ba.block_name:
                    all_qubits.add(gi)
            for q in all_qubits:
                if q in ion_mapping:
                    ion, _ = ion_mapping[q]
                    if not isinstance(ion, SpectatorIon):
                        new_ion_indices.append(ion.idx)
                        new_q2i[q] = ion.idx  # Fix A
            sub_grids[ba.block_name].ion_indices = new_ion_indices
            sub_grids[ba.block_name].qubit_to_ion = new_q2i  # Fix A

        # ── Inject block_sub_grids for ionRoutingGadgetArch ──────────
        circuit.metadata["block_sub_grids"] = sub_grids

        # ── Build QubitMapping ───────────────────────────────────────
        mapping = QubitMapping()
        for stim_idx, (ion, _coords) in ion_mapping.items():
            if not isinstance(ion, SpectatorIon):
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
        _widget_container = None           # ipywidgets container (notebook)

        def _try_notebook_progress(rc):
            """Attempt to upgrade rc.progress_callback to ipywidgets.

            Returns (widget_container, progress_close, did_upgrade).
            On failure or non-notebook environment returns (None, None, False).
            """
            try:
                from .progress_table import (
                    make_notebook_widget_progress_callback,
                    _in_notebook,
                )
                if _in_notebook():
                    from IPython.display import display as _ipy_display
                    container, cb, closer = (
                        make_notebook_widget_progress_callback("WISE Routing")
                    )
                    _ipy_display(container)
                    rc.progress_callback = cb
                    return container, closer, True
            except Exception:
                pass
            return None, None, False

        if self.is_wise and self.wise_config is not None:
            kwargs = dict(self.routing_kwargs)
            if "routing_config" not in kwargs or kwargs.get("routing_config") is None:
                # No routing_config provided — use production defaults
                # (all CPU cores, base_pmax_in=1, tqdm progress).
                kwargs["routing_config"] = WISERoutingConfig.default(
                    show_progress=self.show_progress,
                )
                rc = kwargs["routing_config"]
                if self.show_progress and rc is not None and rc.progress_callback is not None:
                    _widget_container, _progress_close, _ = _try_notebook_progress(rc)
                    if _progress_close is not None:
                        rc._progress_close = _progress_close
                if rc.progress_callback is not None:
                    # Stash the close function so we can clean up later
                    _progress_close = getattr(rc, '_progress_close', None)
            else:
                # routing_config was pre-provided — upgrade progress to
                # ipywidgets in notebooks (even if tqdm was pre-created).
                rc = kwargs["routing_config"]
                if self.show_progress and rc is not None:
                    _widget_container, _progress_close, _did_widget = (
                        _try_notebook_progress(rc)
                    )
                    if not _did_widget and rc.progress_callback is None:
                        # Terminal fallback — create tqdm triple bars
                        _progress_cb, _progress_close = make_triple_tqdm_progress_callback(
                            round_desc="MS Rounds",
                            patch_desc="Patches",
                            sat_desc="SAT Configs",
                        )
                        rc.progress_callback = _progress_cb
                    rc._progress_close = _progress_close
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

            # Detect gadget experiments and dispatch to phase-aware routing
            _qec_meta = circuit.native_circuit.metadata.get("qec_metadata")
            _is_gadget = (
                _qec_meta is not None
                and hasattr(_qec_meta, 'is_gadget')
                and _qec_meta.is_gadget
            )

            if _is_gadget:
                _gadget = circuit.native_circuit.metadata.get("gadget")
                _qubit_alloc = circuit.native_circuit.metadata.get("qubit_allocation")
                _block_sub_grids = circuit.native_circuit.metadata.get("block_sub_grids")

                # ---- Fix 4: Build real qubit→ion mapping ----
                # The analytics path assumes ion.idx == qubit_idx + 1,
                # but compact clustering may assign different indices.
                # Build the actual mapping from the compiler's ion
                # assignment and pass it to ionRoutingGadgetArch so
                # cross-validation (Fix 1) uses the correct index
                # space.
                _ion_map = circuit.native_circuit.metadata.get(
                    "ion_mapping", self._ion_mapping
                )
                _compiler_q2i = {
                    q: ion.idx
                    for q, (ion, _) in _ion_map.items()
                    if not isinstance(ion, SpectatorIon)
                }

                # ---- Fix 2: Build per-toMoveOps phase tags ----
                # Each tag is the *filtered* phase index (init/measure
                # excluded) that the corresponding toMoveOps entry
                # belongs to.  ionRoutingGadgetArch uses these to
                # validate greedy-extraction consistency and skip
                # reordering when tags are available.
                _toMove_phase_tags = None
                _phases_filtered = [
                    p for p in _qec_meta.phases
                    if getattr(p, 'phase_type', '') not in
                       ('init', 'measure', '')
                ]
                if _phases_filtered:
                    _toMove_phase_tags = []
                    for _ph_i, _ph in enumerate(_phases_filtered):
                        _cnt = getattr(_ph, 'ms_pair_count', 0)
                        _toMove_phase_tags.extend([_ph_i] * _cnt)

                # M10: Assert that toMoveOps and _toMove_phase_tags have
                # matching lengths — both represent CX instructions (buckets),
                # NOT individual ion pairs within each bucket.
                if _toMove_phase_tags is not None and _toMoveOps:
                    _expected_len = len(_toMoveOps)
                    if len(_toMove_phase_tags) != _expected_len:
                        logger.warning(
                            "toMove_phase_tags length %d != toMoveOps buckets %d",
                            len(_toMove_phase_tags), _expected_len,
                        )

                allOps, route_barriers, reconfig_time = ionRoutingGadgetArch(
                    arch, self.wise_config, instructions,
                    qec_metadata=_qec_meta,
                    gadget=_gadget,
                    qubit_allocation=_qubit_alloc,
                    block_sub_grids=_block_sub_grids,
                    toMove_phase_tags=_toMove_phase_tags,
                    _compiler_q2i=_compiler_q2i,
                    **kwargs
                )
            else:
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

    def schedule(self, circuit: RoutedCircuit, epoch_mode: str = "edge") -> ScheduledCircuit:
        """Schedule operations into parallel batches.

        Parameters
        ----------
        epoch_mode : str
            How tick-epoch boundaries constrain scheduling:
            ``"edge"`` (default) — same-ion epoch edges only,
            ``"barrier"`` — full barriers between all epoch groups,
            ``"hybrid"`` — barriers only at measurement/reset epochs.

        Returns
        -------
        ScheduledCircuit
            With timing information.
        """
        barriers = circuit.metadata.get("barriers", [])
        is_wise = circuit.metadata.get("is_wise", False)
        allOps = circuit.operations

        # ── Pre-scheduling: reorder rotations for better batching ──
        # Rotations between MS rounds are grouped by type (all RX before
        # all RY) to maximise same-type parallel batches on WISE.
        # Rotations can cross resets/measurements on different qubits
        # but cannot cross MS-round boundaries.
        if is_wise:
            allOps = reorder_rotations_for_batching(list(allOps))
            # Rebuild barriers at type-transition boundaries after
            # reordering.  The original barrier positions (computed on
            # the pre-reorder sequence) become stale when ops move.
            if barriers:
                new_barriers: list[int] = []
                for i in range(1, len(allOps)):
                    prev_op = allOps[i - 1]
                    curr_op = allOps[i]
                    # Barrier ONLY at transport (non-QubitOperation)
                    # boundaries.  MS ↔ rotation type separation is
                    # enforced by the WISE type-matching constraint in
                    # paralleliseOperations, so we do NOT insert barriers
                    # at MS ↔ non-MS transitions — doing so would isolate
                    # each MS gate in a singleton segment and prevent the
                    # scheduler from packing independent MS gates together.
                    if not isinstance(curr_op, QubitOperation) or not isinstance(prev_op, QubitOperation):
                        new_barriers.append(i)
                barriers = sorted(set(new_barriers))

        if barriers:
            parallelOpsMap = paralleliseOperationsWithBarriers(
                allOps, barriers, isWiseArch=is_wise,
                epoch_mode=epoch_mode,
            )
        else:
            parallelOpsMap = paralleliseOperations(
                allOps, isWISEArch=is_wise,
                epoch_mode=epoch_mode,
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
