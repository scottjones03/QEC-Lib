# src/qectostim/experiments/hardware_simulation/trapped_ion/noise.py
"""
Trapped ion noise model.

Hardware-specific noise model for trapped ion quantum computers.
Implements timing-aware noise injection using ExecutionPlan.

Noise sources modeled:
- T2 dephasing during idle time (Z_ERROR)
- Gate infidelity (DEPOLARIZE1/2)
- Gate swap/transport errors (DEPOLARIZE2)
- Measurement errors (X_ERROR)
- MS gate errors dependent on chain length

Reference: hardware_simulation/old/simulator/qccd_circuit.py simulate()
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

from qectostim.noise.hardware.base import (
    HardwareNoiseModel,
    CalibrationData,
    NoiseChannel,
    NoiseChannelType,
)

if TYPE_CHECKING:
    from qectostim.experiments.hardware_simulation.core.operations import PhysicalOperation
    from qectostim.experiments.hardware_simulation.core.execution import (
        ExecutionPlan,
        OperationTiming,
        GateSwapInfo,
    )


@dataclass
class TrappedIonCalibration(CalibrationData):
    """Trapped ion specific calibration data.
    
    Attributes
    ----------
    heating_rate : float
        Motional heating rate in quanta/second.
    ms_infidelity_base : float
        Base MS gate infidelity (before heating effects).
    ms_infidelity_per_ion : float
        Additional infidelity per ion in chain.
    transport_heating : Dict[str, float]
        Heating during transport operations (quanta/operation).
    recool_time : float
        Time for sympathetic re-cooling (μs).
    """
    heating_rate: float = 100.0  # quanta/s
    ms_infidelity_base: float = 0.002
    ms_infidelity_per_ion: float = 0.0005
    transport_heating: Dict[str, float] = None
    recool_time: float = 50.0  # μs
    
    def __post_init__(self):
        if self.transport_heating is None:
            self.transport_heating = {
                "split": 0.5,    # quanta
                "merge": 0.5,
                "shuttle": 0.05,
                "junction": 0.3,
            }


class TrappedIonNoiseModel(HardwareNoiseModel):
    """Noise model for trapped ion hardware.
    
    Models trapped ion specific noise sources:
    - MS gate infidelity (depends on chain length, heating)
    - Motional heating during operations
    - Transport-induced heating (QCCD)
    - Dephasing from T2 decay
    - Measurement errors
    
    NOT IMPLEMENTED: This is a stub defining the interface.
    See hardware_simulation/old/utils/qccd_operations.py for reference.
    
    Parameters
    ----------
    calibration : Optional[TrappedIonCalibration]
        Trapped ion calibration data.
    error_scaling : float
        Scale factor for all errors.
    include_heating : bool
        Whether to model motional heating effects.
    """
    
    def __init__(
        self,
        calibration: Optional[TrappedIonCalibration] = None,
        error_scaling: float = 1.0,
        include_heating: bool = True,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            calibration=calibration or TrappedIonCalibration(),
            error_scaling=error_scaling,
            metadata=metadata,
        )
        self.include_heating = include_heating
    
    @property
    def ion_calibration(self) -> TrappedIonCalibration:
        """Get trapped ion specific calibration."""
        return self.calibration
    
    def apply_to_operation(
        self,
        operation: "PhysicalOperation",
    ) -> List[NoiseChannel]:
        """Get noise channels for a trapped ion operation.
        
        Applies depolarizing noise based on gate fidelity.
        For MS gates, fidelity depends on chain length.
        
        Parameters
        ----------
        operation : PhysicalOperation
            The operation being executed.
            
        Returns
        -------
        List[NoiseChannel]
            Noise channels to apply after the operation.
        """
        channels = []
        qubits = operation.qubits
        
        # Get fidelity from operation or calibration
        fidelity = operation.fidelity(**{}) if hasattr(operation, 'fidelity') else 0.999
        error_prob = (1.0 - fidelity) * self.error_scaling
        error_prob = min(error_prob, 0.5)  # Clamp to valid range
        
        if error_prob <= 0:
            return channels
        
        if len(qubits) == 1:
            # Single-qubit operation
            channels.append(NoiseChannel(
                channel_type=NoiseChannelType.DEPOLARIZING_1Q,
                probability=error_prob,
                qubits=qubits,
            ))
        elif len(qubits) == 2:
            # Two-qubit operation (MS gate)
            channels.append(NoiseChannel(
                channel_type=NoiseChannelType.DEPOLARIZING_2Q,
                probability=error_prob,
                qubits=qubits,
            ))
        
        return channels
    
    def apply_to_operation_timing(
        self,
        timing: "OperationTiming",
    ) -> List[NoiseChannel]:
        """Get noise channels for an operation with timing info.
        
        This is the preferred method when using ExecutionPlan.
        Uses the fidelity from the timing metadata.
        
        Parameters
        ----------
        timing : OperationTiming
            Operation timing with fidelity info.
            
        Returns
        -------
        List[NoiseChannel]
            Noise channels to apply.
        """
        channels = []
        qubits = timing.qubits
        gate_name = timing.gate_name.upper()
        
        # Handle measurement/reset specially
        if gate_name in ("M", "MX", "MY", "MZ", "MR"):
            error_prob = (1.0 - timing.fidelity) * self.error_scaling
            error_prob = min(error_prob, 0.5)
            if error_prob > 0:
                for q in qubits:
                    channels.append(NoiseChannel(
                        channel_type=NoiseChannelType.MEASUREMENT,
                        probability=error_prob,
                        qubits=(q,),
                        stim_instruction=f"X_ERROR({error_prob}) {q}",
                    ))
            return channels
        
        if gate_name in ("R", "RESET"):
            error_prob = (1.0 - timing.fidelity) * self.error_scaling
            error_prob = min(error_prob, 0.5)
            if error_prob > 0:
                for q in qubits:
                    channels.append(NoiseChannel(
                        channel_type=NoiseChannelType.BIT_FLIP,
                        probability=error_prob,
                        qubits=(q,),
                        stim_instruction=f"X_ERROR({error_prob}) {q}",
                    ))
            return channels
        
        # Standard gate noise
        error_prob = (1.0 - timing.fidelity) * self.error_scaling
        error_prob = min(error_prob, 0.5)
        
        if error_prob <= 0:
            return channels
        
        if len(qubits) == 1:
            channels.append(NoiseChannel(
                channel_type=NoiseChannelType.DEPOLARIZING_1Q,
                probability=error_prob,
                qubits=qubits,
            ))
        elif len(qubits) >= 2:
            # Apply to pairs
            for i in range(0, len(qubits) - 1, 2):
                q1, q2 = qubits[i], qubits[i + 1]
                channels.append(NoiseChannel(
                    channel_type=NoiseChannelType.DEPOLARIZING_2Q,
                    probability=error_prob,
                    qubits=(q1, q2),
                ))
        
        return channels
    
    def idle_noise(
        self,
        qubit: int,
        duration: float,
    ) -> List[NoiseChannel]:
        """Get noise for idle time (T2 dephasing).
        
        During idle time, qubits experience dephasing due to T2 decay.
        The error probability follows: p = (1 - exp(-t/T2)) / 2
        
        Parameters
        ----------
        qubit : int
            The idling qubit.
        duration : float
            Idle time in microseconds.
            
        Returns
        -------
        List[NoiseChannel]
            Dephasing noise channel.
        """
        channels = []
        
        if duration <= 0:
            return channels
        
        # Get T2 time from calibration (convert μs to same units)
        # Calibration stores T2 in microseconds
        t2 = self.calibration.get_t2(qubit, default=500000.0)  # 500ms default
        
        # Calculate dephasing probability
        # p = (1 - exp(-t/T2)) / 2
        if t2 > 0:
            p_dephase = (1.0 - np.exp(-duration / t2)) / 2.0
        else:
            p_dephase = 0.0
        
        p_dephase *= self.error_scaling
        p_dephase = min(p_dephase, 0.5)
        
        if p_dephase > 1e-10:  # Skip negligible noise
            channels.append(NoiseChannel(
                channel_type=NoiseChannelType.DEPHASING,
                probability=p_dephase,
                qubits=(qubit,),
                stim_instruction=f"Z_ERROR({p_dephase}) {qubit}",
            ))
        
        return channels
    
    def swap_noise(
        self,
        swap_info: "GateSwapInfo",
    ) -> List[NoiseChannel]:
        """Get noise from ion transport/gate swaps.
        
        When ions need to be co-located for a two-qubit gate,
        transport operations add depolarizing noise.
        
        Parameters
        ----------
        swap_info : GateSwapInfo
            Information about the swap chain.
            
        Returns
        -------
        List[NoiseChannel]
            Depolarizing noise from swaps.
        """
        channels = []
        
        if swap_info.num_swaps <= 0:
            return channels
        
        # Error probability accumulates with each swap
        error_prob = swap_info.error_probability * self.error_scaling
        error_prob = min(error_prob, 0.5)
        
        if error_prob > 0 and len(swap_info.qubits) >= 2:
            q1, q2 = swap_info.qubits[0], swap_info.qubits[1]
            channels.append(NoiseChannel(
                channel_type=NoiseChannelType.DEPOLARIZING_2Q,
                probability=error_prob,
                qubits=(q1, q2),
                stim_instruction=f"DEPOLARIZE2({error_prob}) {q1} {q2}",
            ))
        
        return channels
    
    def apply_with_plan(
        self,
        circuit: stim.Circuit,
        plan: "ExecutionPlan",
    ) -> stim.Circuit:
        """Apply timing-aware noise using execution plan.
        
        This overrides the base class to use trapped-ion-specific
        noise injection with proper ordering:
        1. Idle dephasing (Z_ERROR)
        2. Gate swap noise (DEPOLARIZE2)
        3. Original instruction
        4. Gate infidelity noise (DEPOLARIZE1/2)
        
        Parameters
        ----------
        circuit : stim.Circuit
            The Stim circuit.
        plan : ExecutionPlan
            Execution plan with timing metadata.
            
        Returns
        -------
        stim.Circuit
            Circuit with noise applied.
        """
        from qectostim.experiments.hardware_simulation.core.execution import (
            ExecutionPlan,
            OperationTiming,
        )
        
        noisy = stim.Circuit()
        instruction_index = 0
        
        for inst in circuit.flattened():
            name = inst.name.upper()
            
            # Skip annotations - just append them
            if name in {"TICK", "DETECTOR", "OBSERVABLE_INCLUDE", 
                        "SHIFT_COORDS", "QUBIT_COORDS", "BARRIER"}:
                noisy.append(inst)
                continue
            
            qubit_targets = tuple(
                t.value for t in inst.targets_copy() if t.is_qubit_target
            )
            
            if not qubit_targets:
                noisy.append(inst)
                continue
            
            # 1. Idle dephasing BEFORE this instruction
            for idle in plan.get_idle_before(instruction_index):
                for channel in self.idle_noise(idle.qubit, idle.duration):
                    if channel.probability > 0:
                        noisy.append_from_stim_program_text(channel.to_stim())
            
            # 2. Gate swap noise (if any)
            for swap_info in plan.get_swaps_for_instruction(instruction_index):
                for channel in self.swap_noise(swap_info):
                    if channel.probability > 0:
                        noisy.append_from_stim_program_text(channel.to_stim())
            
            # 3. Original instruction
            noisy.append(inst)
            
            # 4. Gate infidelity noise AFTER
            timing = plan.get_operation(instruction_index)
            if timing is not None:
                for channel in self.apply_to_operation_timing(timing):
                    if channel.probability > 0:
                        noisy.append_from_stim_program_text(channel.to_stim())
            
            instruction_index += 1
        
        return noisy
    
    def ms_gate_fidelity(
        self,
        chain_length: int,
        motional_quanta: float = 0.0,
    ) -> float:
        """Calculate MS gate fidelity.
        
        Fidelity decreases with:
        - Longer chains (more spectator modes)
        - Higher motional excitation (from heating)
        
        NOT IMPLEMENTED: Returns placeholder.
        """
        # Placeholder formula from old code
        base = 1.0 - self.ion_calibration.ms_infidelity_base
        chain_penalty = self.ion_calibration.ms_infidelity_per_ion * chain_length
        heating_penalty = 0.001 * motional_quanta  # Approximate
        
        fidelity = base - chain_penalty - heating_penalty
        return max(0.0, fidelity * self.error_scaling)
    
    def transport_error(
        self,
        operation_type: str,
    ) -> float:
        """Get error from transport operation.
        
        NOT IMPLEMENTED.
        """
        quanta = self.ion_calibration.transport_heating.get(operation_type, 0.1)
        # Simple model: error proportional to heating
        return quanta * 0.01 * self.error_scaling
