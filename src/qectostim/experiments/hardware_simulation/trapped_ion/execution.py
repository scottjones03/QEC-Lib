# src/qectostim/experiments/hardware_simulation/trapped_ion/execution.py
"""
Trapped ion execution planner.

Creates ExecutionPlan from compilation metadata for timing-aware noise injection.
This extracts timing, routing, and gate swap information from the compiler's
output WITHOUT modifying the Stim circuit.

The planner bridges between:
- The abstract Stim circuit (instruction indices)
- The physical compilation (timing, routing, zones)

Usage:
    planner = TrappedIonExecutionPlanner(compiler, calibration)
    plan = planner.plan_execution(circuit, compiled)
    noisy_circuit = noise_model.apply_with_plan(circuit, plan)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import (
    Dict,
    List,
    Optional,
    Tuple,
    Any,
    TYPE_CHECKING,
)

import stim

from qectostim.experiments.hardware_simulation.core.execution import (
    ExecutionPlan,
    OperationTiming,
    GateSwapInfo,
    IdleInterval,
)

from qectostim.experiments.hardware_simulation.trapped_ion.physics import (
    IonChainFidelityModel,
    DEFAULT_FIDELITY_MODEL as _FIDELITY,
    DEFAULT_CALIBRATION as _CALIBRATION,
)

if TYPE_CHECKING:
    from qectostim.experiments.hardware_simulation.core.pipeline import (
        CompiledCircuit,
        ScheduledCircuit,
        ScheduledOperation,
    )
    from qectostim.experiments.hardware_simulation.trapped_ion.compiler import (
        TrappedIonCompiler,
    )
    from qectostim.experiments.hardware_simulation.trapped_ion.noise import (
        TrappedIonCalibration,
    )
    from qectostim.noise.hardware.base import CalibrationData


# Gate times derived from CalibrationConstants (physics.py).
# Values are in **microseconds**.
DEFAULT_GATE_TIMES: Dict[str, float] = _CALIBRATION.gate_times_us()

# Gate fidelities derived from CalibrationConstants for a default chain of 2.
DEFAULT_FIDELITIES: Dict[str, float] = _CALIBRATION.gate_fidelities(chain_length=2)

_logger = logging.getLogger(__name__)


class TrappedIonExecutionPlanner:
    """Creates ExecutionPlan for trapped ion hardware.
    
    Extracts timing and routing information from compilation to enable
    timing-aware noise injection. This does NOT modify the circuit -
    it only computes metadata.
    
    Parameters
    ----------
    compiler : TrappedIonCompiler
        The compiler used for this circuit.
    calibration : Optional[CalibrationData]
        Hardware calibration data for fidelity lookup.
    gate_times : Optional[Dict[str, float]]
        Override gate times (μs). Uses defaults if not provided.
    """
    
    def __init__(
        self,
        compiler: Optional["TrappedIonCompiler"] = None,
        calibration: Optional["CalibrationData"] = None,
        gate_times: Optional[Dict[str, float]] = None,
    ):
        self.compiler = compiler
        self.calibration = calibration
        self.gate_times = gate_times or DEFAULT_GATE_TIMES.copy()
        self._default_fidelities = DEFAULT_FIDELITIES.copy()
    
    def plan_execution(
        self,
        circuit: stim.Circuit,
        compiled: Optional["CompiledCircuit"] = None,
    ) -> ExecutionPlan:
        """Create execution plan from circuit and compilation result.
        
        Parameters
        ----------
        circuit : stim.Circuit
            The Stim circuit to plan execution for.
        compiled : Optional[CompiledCircuit]
            Compilation result with scheduling info. If None, estimates
            timing from gate times.
            
        Returns
        -------
        ExecutionPlan
            Execution plan for noise injection.
        """
        plan = ExecutionPlan()
        
        # Track per-qubit timing for idle calculation
        qubit_last_end_time: Dict[int, float] = {}
        current_time = 0.0
        instruction_index = 0
        
        # Flatten the circuit to get instruction order
        for inst in circuit.flattened():
            name = inst.name.upper()
            
            # Skip annotations and barriers
            if name in {"TICK", "DETECTOR", "OBSERVABLE_INCLUDE", 
                        "SHIFT_COORDS", "QUBIT_COORDS", "BARRIER"}:
                continue
            
            # Extract qubits
            qubit_targets = tuple(
                t.value for t in inst.targets_copy() if t.is_qubit_target
            )
            
            if not qubit_targets:
                continue
            
            # Get timing from compiled circuit or estimate
            if compiled is not None and hasattr(compiled, 'scheduled'):
                timing = self._get_timing_from_compiled(
                    instruction_index, name, qubit_targets, compiled
                )
            else:
                timing = self._estimate_timing(
                    instruction_index, name, qubit_targets, current_time
                )
            
            # Calculate idle intervals for involved qubits
            for qubit in qubit_targets:
                last_end = qubit_last_end_time.get(qubit, 0.0)
                if timing.start_time > last_end:
                    # There's an idle gap
                    idle = IdleInterval(
                        qubit=qubit,
                        start_time=last_end,
                        end_time=timing.start_time,
                        following_instruction=instruction_index,
                    )
                    plan.idle_intervals.append(idle)
                qubit_last_end_time[qubit] = timing.end_time
            
            # Add gate swap info if this was a two-qubit gate requiring routing
            if compiled is not None and len(qubit_targets) >= 2:
                swap_info = self._get_swap_info(instruction_index, qubit_targets, compiled)
                if swap_info is not None:
                    plan.gate_swaps.append(swap_info)
            
            plan.operations.append(timing)
            current_time = max(current_time, timing.end_time)
            instruction_index += 1
        
        # Finalize plan
        plan.total_duration = current_time
        plan.num_qubits = max(qubit_last_end_time.keys(), default=-1) + 1
        
        return plan
    
    def _get_timing_from_compiled(
        self,
        instruction_index: int,
        gate_name: str,
        qubits: Tuple[int, ...],
        compiled: "CompiledCircuit",
    ) -> OperationTiming:
        """Extract timing from CompiledCircuit's scheduled operations.

        Uses the stim_instruction_map (when available) to determine how
        many native ops each stim instruction decomposes into, which
        scales the idle-dephasing duration in the noise model.

        Timing (start_time, duration) is estimated sequentially from
        the scheduled_ops list, using the instruction_index to look up
        the corresponding scheduled operation.

        Fidelity is computed using the physics formula with the n̄
        value at the time the gate executes (per-batch snapshot).
        The motional_quanta and chain_length are stored on the
        returned OperationTiming so the noise model can recompute
        fidelity dynamically if needed.
        """
        scheduled = compiled.scheduled
        duration = self.gate_times.get(
            gate_name, _CALIBRATION.single_qubit_gate_time * 1e6
        )

        # Get per-gate n̄ and mode snapshot from per-batch snapshot
        chain_length, motional_quanta, batch_index, mode_snapshot = (
            self._extract_chain_and_heating(qubits, compiled)
        )
        fidelity = self._get_fidelity(
            gate_name, qubits,
            chain_length=chain_length,
            motional_quanta=motional_quanta,
        )

        # --- Determine num_native_ops from stim_instruction_map ----------
        num_native_ops = 1
        native_circuit = None
        if hasattr(compiled, 'scheduled') and compiled.scheduled.routed_circuit is not None:
            routed = compiled.scheduled.routed_circuit
            if hasattr(routed, 'mapped_circuit') and routed.mapped_circuit is not None:
                native_circuit = routed.mapped_circuit.native_circuit

        if (native_circuit is not None
                and hasattr(native_circuit, 'stim_instruction_map')
                and native_circuit.stim_instruction_map):
            sim = native_circuit.stim_instruction_map
            gate_keys = sorted(sim.keys())
            if instruction_index < len(gate_keys):
                native_idxs = sim[gate_keys[instruction_index]]
                num_native_ops = max(1, len(native_idxs))

        # --- Timing from scheduled_ops ----------------------------------
        start_time = 0.0
        if scheduled.scheduled_ops and instruction_index < len(scheduled.scheduled_ops):
            sched_op = scheduled.scheduled_ops[instruction_index]
            start_time = sched_op.start_time
            duration = sched_op.duration

        return OperationTiming(
            instruction_index=instruction_index,
            gate_name=gate_name,
            qubits=qubits,
            start_time=start_time,
            duration=duration,
            fidelity=fidelity,
            batch_index=batch_index,
            num_native_ops=num_native_ops,
            platform_context={
                "chain_length": chain_length,
                "motional_quanta": motional_quanta,
                "mode_snapshot": mode_snapshot,
            },
        )
    
    def _extract_chain_and_heating(
        self,
        qubits: Tuple[int, ...],
        compiled: "CompiledCircuit",
    ) -> Tuple[Optional[int], float, Optional[int], Optional[Any]]:
        """Extract chain_length, motional_quanta, and mode snapshot.

        Returns the n̄ at the time the gate executes, NOT the
        end-of-circuit accumulated total.

        Strategy:
        1. Look up which batch this gate pair belongs to using
           ``gate_batch_map``.
        2. Retrieve the per-batch n̄ snapshot from
           ``motional_quanta_per_batch[batch_index]``.
        3. Compute trap-level motional quanta as
           ``per_ion_nbar * chain_length`` (matching old code's
           ``trap.motionalMode = sum(ion.motionalMode for ion)``).
        4. If ``mode_snapshots_per_batch`` is available, retrieve the
           ModeSnapshot for this gate's qubits — this carries the full
           3N normal-mode frequencies, eigenvectors, and per-mode
           occupancies that the collaborator's noise model will consume.

        If per-batch snapshots are not available (e.g. no routing was
        performed), falls back to the end-of-circuit values.

        Returns ``(chain_length, motional_quanta, batch_index, mode_snapshot)``.
        """
        chain_length: Optional[int] = None
        motional_quanta: float = 0.0
        batch_index: Optional[int] = None
        mode_snapshot: Optional[Any] = None

        comp_meta = compiled.metrics or {}

        # --- Chain length ---
        if "chain_lengths" in comp_meta:
            cl_map = comp_meta["chain_lengths"]
            for q in qubits:
                if q in cl_map:
                    chain_length = cl_map[q]
                    break
        if chain_length is None and "default_chain_length" in comp_meta:
            chain_length = comp_meta["default_chain_length"]

        n_ions = chain_length if chain_length and chain_length >= 1 else 1

        # --- Per-batch n̄ (preferred: exact system state at gate time) ---
        per_batch = comp_meta.get("motional_quanta_per_batch")
        batch_map = comp_meta.get("gate_batch_map")

        if per_batch and batch_map is not None:
            # Find the batch index for this gate pair
            pair = (qubits[0], qubits[1]) if len(qubits) >= 2 else None
            if pair and pair in batch_map:
                batch_index = batch_map[pair]
            elif pair and (pair[1], pair[0]) in batch_map:
                batch_index = batch_map[(pair[1], pair[0])]

            if batch_index is not None and batch_index < len(per_batch):
                snapshot = per_batch[batch_index]
                # Per-ion n̄ for one of the involved qubits
                per_ion = 0.0
                for q in qubits:
                    if q in snapshot:
                        per_ion = snapshot[q]
                        break
                else:
                    if snapshot:
                        per_ion = sum(snapshot.values()) / len(snapshot)
                # Trap-level = per_ion × chain_length
                motional_quanta = per_ion * n_ions

                # --- Mode snapshot (3N normal-mode state) ---
                # The routing pass captured a snapshot of each trap's
                # full vibrational state (mode frequencies, eigenvectors,
                # per-mode occupancies) AFTER transport heating but
                # BEFORE this gate executes.  We look up the snapshot
                # for the specific qubit involved in this gate.
                #
                # This is the key data the collaborator's noise model
                # needs: it tells them "these are the 3N mode frequencies
                # and occupancies right before your gate runs" — enabling
                # mode-resolved, potentially correlated error computation.
                mode_per_batch = comp_meta.get("mode_snapshots_per_batch")
                if mode_per_batch and batch_index < len(mode_per_batch):
                    mode_batch = mode_per_batch[batch_index]
                    for q in qubits:
                        if q in mode_batch:
                            mode_snapshot = mode_batch[q]
                            break

                return chain_length, motional_quanta, batch_index, mode_snapshot

        # --- Fallback: end-of-circuit n̄ (when no per-batch data) ---
        if "motional_quanta" in comp_meta:
            mq_map = comp_meta["motional_quanta"]
            per_ion = 0.0
            for q in qubits:
                if q in mq_map:
                    per_ion = mq_map[q]
                    break
            else:
                if mq_map:
                    per_ion = sum(mq_map.values()) / len(mq_map)
            motional_quanta = per_ion * n_ions

        return chain_length, motional_quanta, batch_index, mode_snapshot
    
    def _estimate_timing(
        self,
        instruction_index: int,
        gate_name: str,
        qubits: Tuple[int, ...],
        current_time: float,
    ) -> OperationTiming:
        duration = self.gate_times.get(
            gate_name, _CALIBRATION.single_qubit_gate_time * 1e6
        )
        # When estimating, we don't know chain_length, fall back to defaults
        fidelity = self._get_fidelity(gate_name, qubits)
        
        return OperationTiming(
            instruction_index=instruction_index,
            gate_name=gate_name,
            qubits=qubits,
            start_time=current_time,
            duration=duration,
            fidelity=fidelity,
        )
    
    def _get_fidelity(
        self,
        gate_name: str,
        qubits: Tuple[int, ...],
        chain_length: Optional[int] = None,
        motional_quanta: float = 0.0,
    ) -> float:
        """Get gate fidelity from calibration or defaults.

        Delegates to :class:`IonChainFidelityModel` in physics.py
        when ``chain_length`` is known.  Otherwise falls back to
        fixed lookup tables.
        """
        # Try physics formula
        if chain_length is not None and chain_length >= 1:
            TWO_QUBIT_GATES = {
                "MS", "CX", "CZ", "CNOT", "SWAP", "ISWAP",
                "XCX", "XCZ", "YCX", "ZCX", "ZCZ",
            }
            is_2q = gate_name in TWO_QUBIT_GATES
            return _FIDELITY.gate_fidelity(
                chain_length, motional_quanta, is_two_qubit=is_2q,
            )

        # Fixed calibration lookup
        if self.calibration is not None:
            if len(qubits) == 1:
                return self.calibration.get_1q_fidelity(qubits[0], gate_name)
            elif len(qubits) == 2:
                return self.calibration.get_2q_fidelity(qubits[0], qubits[1], gate_name)

        # Fall back to defaults
        return self._default_fidelities.get(
            gate_name, 1.0 - _CALIBRATION.measurement_infidelity
        )
    
    def _get_swap_info(
        self,
        instruction_index: int,
        qubits: Tuple[int, ...],
        compiled: "CompiledCircuit",
    ) -> Optional[GateSwapInfo]:
        """Extract gate swap info from compilation.
        
        For QCCD architectures, two-qubit gates may require ion transport.
        This extracts that information from the routed circuit metadata,
        the compiled metrics dict, or the routed_circuit's routing_operations.
        
        Gate swap fidelity is computed as MS_fidelity^3 to match the old
        code's GateSwap.calculateFidelity() which uses 3 MS gates per swap.
        """
        pair = (qubits[0], qubits[1]) if len(qubits) >= 2 else qubits

        # Get chain length and motional quanta for fidelity calculation
        chain_length, motional_quanta, _batch_idx, mode_snapshot = self._extract_chain_and_heating(qubits, compiled)
        
        # Compute MS gate fidelity using physics formula
        ms_fidelity = self._get_fidelity(
            "MS", qubits, chain_length, motional_quanta
        )
        
        # GateSwap = 3 MS gates (old code: GateSwap.calculateFidelity = product of 3)
        swap_fidelity = ms_fidelity ** 3

        # 1. Check compiled.metrics["routing_swaps"]
        if compiled.metrics and "routing_swaps" in compiled.metrics:
            routing_swaps = compiled.metrics["routing_swaps"]
            if instruction_index in routing_swaps:
                swap_data = routing_swaps[instruction_index]
                return GateSwapInfo(
                    instruction_index=instruction_index,
                    qubits=pair,
                    num_swaps=swap_data.get("num_swaps", 0),
                    swap_fidelity=swap_fidelity,
                    transport_time=swap_data.get("time", 0.0),
                    metadata={
                        "ms_fidelity": ms_fidelity,
                        "chain_length": chain_length,
                        "motional_quanta": motional_quanta,
                    },
                    platform_context={
                        "chain_length": chain_length,
                        "motional_quanta": motional_quanta,
                        "mode_snapshot": mode_snapshot,
                    },
                )

        # 2. Check routed_circuit.metadata["gate_swaps_by_stim_idx"]
        routed = (
            compiled.scheduled.routed_circuit
            if compiled.scheduled and compiled.scheduled.routed_circuit
            else None
        )
        if routed is not None and routed.metadata:
            swaps_by_idx = routed.metadata.get("gate_swaps_by_stim_idx", {})
            if instruction_index in swaps_by_idx:
                swap_list = swaps_by_idx[instruction_index]
                total_swaps = sum(
                    getattr(s, 'num_swaps', 1) for s in swap_list
                )
                return GateSwapInfo(
                    instruction_index=instruction_index,
                    qubits=pair,
                    num_swaps=total_swaps,
                    swap_fidelity=swap_fidelity,
                    transport_time=0.0,
                    metadata={
                        "ms_fidelity": ms_fidelity,
                        "chain_length": chain_length,
                        "motional_quanta": motional_quanta,
                    },
                    platform_context={
                        "chain_length": chain_length,
                        "motional_quanta": motional_quanta,
                        "mode_snapshot": mode_snapshot,
                    },
                )

        # 3. Count routing_operations that touch these qubits
        if routed is not None and routed.routing_operations:
            n_swaps = 0
            for rop in routed.routing_operations:
                rop_qubits = getattr(rop, 'qubits', ())
                if set(rop_qubits) & set(pair):
                    n_swaps += 1
            if n_swaps > 0:
                return GateSwapInfo(
                    instruction_index=instruction_index,
                    qubits=pair,
                    num_swaps=n_swaps,
                    swap_fidelity=swap_fidelity,
                    transport_time=0.0,
                    metadata={
                        "ms_fidelity": ms_fidelity,
                        "chain_length": chain_length,
                        "motional_quanta": motional_quanta,
                    },
                    platform_context={
                        "chain_length": chain_length,
                        "motional_quanta": motional_quanta,
                        "mode_snapshot": mode_snapshot,
                    },
                )

        # No routing info available
        return None


def create_simple_execution_plan(
    circuit: stim.Circuit,
    gate_times: Optional[Dict[str, float]] = None,
    gate_fidelities: Optional[Dict[str, float]] = None,
) -> ExecutionPlan:
    """Create a simple execution plan without compilation info.
    
    This is useful for testing or when using the old noise model
    with timing estimates.
    
    Parameters
    ----------
    circuit : stim.Circuit
        The Stim circuit.
    gate_times : Optional[Dict[str, float]]
        Gate durations in microseconds.
    gate_fidelities : Optional[Dict[str, float]]
        Gate fidelities.
        
    Returns
    -------
    ExecutionPlan
        Simple execution plan with estimated timing.
    """
    planner = TrappedIonExecutionPlanner(
        gate_times=gate_times,
    )
    # Override fidelities if provided
    if gate_fidelities is not None:
        planner._default_fidelities.update(gate_fidelities)
    
    return planner.plan_execution(circuit, compiled=None)
