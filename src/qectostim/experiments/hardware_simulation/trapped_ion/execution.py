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

from dataclasses import dataclass, field
from typing import (
    Dict,
    List,
    Optional,
    Tuple,
    Any,
    TYPE_CHECKING,
)

import numpy as np
import stim

from qectostim.experiments.hardware_simulation.core.execution import (
    ExecutionPlan,
    OperationTiming,
    GateSwapInfo,
    IdleInterval,
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


# Default gate times for trapped ions (microseconds)
DEFAULT_GATE_TIMES = {
    "MS": 100.0,      # Mølmer-Sørensen gate
    "RX": 10.0,       # X rotation
    "RY": 10.0,       # Y rotation  
    "RZ": 0.1,        # Z rotation (virtual, nearly instant)
    "H": 10.0,        # Hadamard (decomposed)
    "CNOT": 250.0,    # CNOT (2 MS + rotations)
    "CX": 250.0,      # Same as CNOT
    "CZ": 270.0,      # CZ (CNOT + H gates)
    "M": 200.0,       # Measurement
    "R": 50.0,        # Reset
    "MR": 250.0,      # Measure + reset
    "SWAP": 750.0,    # SWAP (3 CNOTs)
}

# Default fidelities
DEFAULT_FIDELITIES = {
    "MS": 0.995,
    "RX": 0.9999,
    "RY": 0.9999,
    "RZ": 0.99999,
    "H": 0.9998,
    "CNOT": 0.99,
    "CX": 0.99,
    "CZ": 0.985,
    "M": 0.995,
    "R": 0.999,
    "SWAP": 0.97,
}


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
        """Extract timing from CompiledCircuit's scheduled operations."""
        scheduled = compiled.scheduled
        
        # Try to find matching scheduled operation
        # Note: The scheduled circuit may have different indexing due to decomposition
        # For now, estimate timing based on layer structure
        start_time = 0.0
        duration = self.gate_times.get(gate_name, 10.0)
        
        # If we have scheduled ops, use their timing
        if scheduled.scheduled_ops and instruction_index < len(scheduled.scheduled_ops):
            sched_op = scheduled.scheduled_ops[instruction_index]
            start_time = sched_op.start_time
            duration = sched_op.duration
        
        # Get fidelity from calibration or defaults
        fidelity = self._get_fidelity(gate_name, qubits)
        
        return OperationTiming(
            instruction_index=instruction_index,
            gate_name=gate_name,
            qubits=qubits,
            start_time=start_time,
            duration=duration,
            fidelity=fidelity,
        )
    
    def _estimate_timing(
        self,
        instruction_index: int,
        gate_name: str,
        qubits: Tuple[int, ...],
        current_time: float,
    ) -> OperationTiming:
        """Estimate timing when no compilation info available."""
        duration = self.gate_times.get(gate_name, 10.0)
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
    ) -> float:
        """Get gate fidelity from calibration or defaults."""
        if self.calibration is not None:
            if len(qubits) == 1:
                return self.calibration.get_1q_fidelity(qubits[0], gate_name)
            elif len(qubits) == 2:
                return self.calibration.get_2q_fidelity(qubits[0], qubits[1], gate_name)
        
        # Fall back to defaults
        return self._default_fidelities.get(gate_name, 0.999)
    
    def _get_swap_info(
        self,
        instruction_index: int,
        qubits: Tuple[int, ...],
        compiled: "CompiledCircuit",
    ) -> Optional[GateSwapInfo]:
        """Extract gate swap info from compilation.
        
        For QCCD architectures, two-qubit gates may require ion transport.
        This extracts that information from the routed circuit.
        """
        # Check if routing info is available in metadata
        if compiled.metrics and "routing_swaps" in compiled.metrics:
            routing_swaps = compiled.metrics["routing_swaps"]
            if instruction_index in routing_swaps:
                swap_data = routing_swaps[instruction_index]
                return GateSwapInfo(
                    instruction_index=instruction_index,
                    qubits=(qubits[0], qubits[1]) if len(qubits) >= 2 else qubits,
                    num_swaps=swap_data.get("num_swaps", 0),
                    swap_fidelity=swap_data.get("fidelity", 0.998),
                    transport_time=swap_data.get("time", 0.0),
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
