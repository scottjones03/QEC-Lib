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
    Callable,
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

from qectostim.experiments.hardware_simulation.trapped_ion.physics import (
    CalibrationConstants,
    DEFAULT_CALIBRATION,
    IonChainFidelityModel,
    DEFAULT_FIDELITY_MODEL,
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

    Thin wrapper around :class:`CalibrationConstants` from physics.py
    that also satisfies the :class:`CalibrationData` interface required
    by the noise pipeline.

    All default values are read from ``DEFAULT_CALIBRATION`` so there
    is exactly one source of truth for the numbers.
    """
    # --- Background heating (for gate fidelity formula) ---
    heating_rate: float = DEFAULT_CALIBRATION.heating_rate

    ms_infidelity_base: float = 0.002   # legacy, unused with physics formula
    ms_infidelity_per_ion: float = 0.0005  # legacy, unused
    transport_heating: Dict[str, float] = None
    recool_time: float = DEFAULT_CALIBRATION.recool_time

    # --- Physics-formula constants (from CalibrationConstants) ---
    fidelity_scaling_A: float = DEFAULT_CALIBRATION.fidelity_scaling_A
    ms_gate_time: float = DEFAULT_CALIBRATION.ms_gate_time
    single_qubit_gate_time: float = DEFAULT_CALIBRATION.single_qubit_gate_time
    t2_time: float = DEFAULT_CALIBRATION.t2_time
    measurement_time: float = DEFAULT_CALIBRATION.measurement_time
    reset_time: float = DEFAULT_CALIBRATION.reset_time
    measurement_infidelity: float = DEFAULT_CALIBRATION.measurement_infidelity
    reset_infidelity: float = DEFAULT_CALIBRATION.reset_infidelity

    def __post_init__(self):
        if self.transport_heating is None:
            # Delegate to CalibrationConstants for the default table
            self.transport_heating = dict(DEFAULT_CALIBRATION.transport_heating)


class TrappedIonNoiseModel(HardwareNoiseModel):
    """Noise model for trapped ion QCCD hardware.

    Models trapped-ion-specific noise sources:

    * **Gate infidelity** — Both single-qubit and MS (two-qubit) gates
      use the same physics formula from arXiv:2004.04706 (page 7):
      ``F = 1 - (heating_rate * t_gate + A * N/ln(N) * (2n_bar + 1))``.
      The only difference is the gate time: 5 us (1Q) vs 40 us (MS).
    * **Gate-swap (SWAP) errors** — Each SWAP decomposes into 3 MS gates
      (Fig. 5, arXiv:2004.04706); fidelity is the product of 3 MS
      fidelities at the current chain length and motional quanta.
    * **Measurement error** — ``X_ERROR(1e-3)`` before ``M`` (Table IV,
      PhysRevA.99.022330).
    * **Reset error** — ``X_ERROR(5e-3)`` before ``R`` (Table IV,
      PhysRevA.99.022330).
    * **T2 dephasing** — ``Z_ERROR`` during idle intervals with
      ``p = (1 - exp(-t/T2))/2``, T2 = 2.2 s (PhysRevA.99.022330).
    * **Transport heating** — Depolarising noise from ion shuttling /
      split / merge / junction crossing in QCCD architectures.

    Parameters
    ----------
    calibration : Optional[TrappedIonCalibration]
        Trapped ion calibration data.
    error_scaling : float
        Scale factor for all errors (1.0 = physical noise).
    include_heating : bool
        Whether to model motional heating effects.
    mode_noise_callback : Optional[Callable]
        Optional callback for a correlated/mode-aware noise model.

        When the 3N normal-mode tracking infrastructure is active, every
        ``OperationTiming`` and ``GateSwapInfo`` carries a ``ModeSnapshot``
        that captures the full modal state of the ion chain at the
        instant a gate executes (mode frequencies, eigenvectors, per-mode
        occupancies).

        Setting this callback lets a collaborator's noise model consume
        that information **without** modifying the existing scalar
        fidelity pipeline.  The callback signature is::

            def my_noise_fn(
                gate_name: str,
                qubits: Tuple[int, ...],
                mode_snapshot: ModeSnapshot,   # from architecture.py
                current_channels: List[NoiseChannel],
            ) -> List[NoiseChannel]:
                ...

        The callback receives the standard noise channels produced by
        the scalar model and may *replace*, *augment*, or pass them
        through unchanged.  If ``mode_snapshot`` is ``None`` (e.g. when
        running without the mode tracker), the callback is not invoked.

        Example — replace depolarising error with a correlated model::

            def correlated_noise(gate, qubits, snap, channels):
                freqs = snap.mode_frequencies   # shape (3N,)
                vecs  = snap.eigenvectors       # shape (3N, N)
                occ   = snap.occupancies        # shape (3N,)
                p_err = my_correlated_model(freqs, vecs, occ, gate)
                return [NoiseChannel(..., probability=p_err, ...)]

            model = TrappedIonNoiseModel(mode_noise_callback=correlated_noise)
    """
    
    def __init__(
        self,
        calibration: Optional[TrappedIonCalibration] = None,
        error_scaling: float = 1.0,
        include_heating: bool = True,
        metadata: Optional[Dict[str, Any]] = None,
        mode_noise_callback: Optional[Callable] = None,
    ):
        super().__init__(
            calibration=calibration or TrappedIonCalibration(),
            error_scaling=error_scaling,
            metadata=metadata,
        )
        self.include_heating = include_heating
        self.mode_noise_callback = mode_noise_callback
        # Cache a single IonChainFidelityModel built from our calibration,
        # instead of creating a fresh one on every fidelity call.
        self._fidelity_model = IonChainFidelityModel(
            CalibrationConstants(
                heating_rate=self.ion_calibration.heating_rate,
                fidelity_scaling_A=self.ion_calibration.fidelity_scaling_A,
                ms_gate_time=self.ion_calibration.ms_gate_time,
                single_qubit_gate_time=self.ion_calibration.single_qubit_gate_time,
                t2_time=self.ion_calibration.t2_time,
            )
        )
    
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
        error_prob = (1.0 - fidelity) / self.error_scaling
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
        
        When ``timing.chain_length`` is available, dynamically
        recomputes fidelity from the physics formula using the
        per-gate ``timing.motional_quanta`` (the system state *at
        the time the gate executes*), overriding any pre-baked
        ``timing.fidelity``.  This ensures that gate 1 in batch 0
        sees a different n̄ than gate 5 in batch 3.
        
        Parameters
        ----------
        timing : OperationTiming
            Operation timing with per-gate motional quanta.
            
        Returns
        -------
        List[NoiseChannel]
            Noise channels to apply.
        """
        channels = []
        qubits = timing.qubits
        gate_name = timing.gate_name.upper()
        
        # Handle measurement/reset specially.
        # Use calibration defaults when timing.fidelity is unset (1.0),
        # matching the old ground-truth Measurement.INFIDELITY = 1e-3
        # and QubitReset.INFIDELITY = 5e-3 from [2] Table IV.
        if gate_name in ("M", "MX", "MY", "MZ", "MR"):
            meas_fidelity = timing.fidelity
            if meas_fidelity >= 1.0:
                # Fallback: use calibration default (1e-3 from [2] Table IV)
                meas_fidelity = 1.0 - self.ion_calibration.measurement_infidelity
            error_prob = (1.0 - meas_fidelity) / self.error_scaling
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
            reset_fidelity = timing.fidelity
            if reset_fidelity >= 1.0:
                # Fallback: use calibration default (5e-3 from [2] Table IV)
                reset_fidelity = 1.0 - self.ion_calibration.reset_infidelity
            error_prob = (1.0 - reset_fidelity) / self.error_scaling
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
        
        # --- Dynamic fidelity recomputation ---
        # If the timing carries per-gate motional_quanta and chain_length,
        # recompute fidelity from the physics formula instead of using
        # the pre-baked timing.fidelity.  This is the critical fix: each
        # gate uses the exact system state at the time it executes.
        fidelity = timing.fidelity
        _pctx = timing.platform_context or {}
        _chain_length = _pctx.get("chain_length", None)
        _motional_q = _pctx.get("motional_quanta", None)
        if _chain_length is not None and _chain_length >= 1:
            TWO_QUBIT_GATES = {
                "MS", "CX", "CZ", "CNOT", "SWAP", "ISWAP",
                "XCX", "XCZ", "YCX", "ZCX", "ZCZ",
            }
            if gate_name in TWO_QUBIT_GATES:
                fidelity = self.ms_gate_fidelity(
                    _chain_length, _motional_q
                )
            else:
                fidelity = self.single_qubit_gate_fidelity(
                    _chain_length, _motional_q
                )

        # Standard gate noise
        error_prob = (1.0 - fidelity) / self.error_scaling
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
        
        # --- Dual-path noise architecture ---
        # At this point, `channels` contains the standard scalar noise
        # (DEPOLARIZE from the existing F = 1 − (ṅ·t + A·N/ln(N)·(2n̄+1))
        # formula).  If a mode_noise_callback is registered AND the
        # pipeline has a mode snapshot for this gate, we hand the
        # collaborator's model the full 3N mode state (frequencies,
        # eigenvectors, per-mode occupancies) so they can compute
        # mode-resolved, potentially correlated errors.
        #
        # The callback may replace, augment, or pass through the
        # existing channels — the architecture is non-destructive
        # until the collaborator is ready.
        mode_snapshot = _pctx.get("mode_snapshot", None)
        if self.mode_noise_callback is not None and mode_snapshot is not None:
            channels = self.mode_noise_callback(
                gate_name, qubits, mode_snapshot, channels,
            )

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
        
        # Get T2 time from calibration (in microseconds).
        # Default 2.2s = 2_200_000 μs from PhysRevA.99.022330.
        t2 = self.calibration.get_t2(qubit, default=2_200_000.0)  # 2.2s default
        
        # Calculate dephasing probability
        # p = (1 - exp(-t/T2)) / 2
        if t2 > 0:
            p_dephase = (1.0 - np.exp(-duration / t2)) / 2.0
        else:
            p_dephase = 0.0
        
        p_dephase /= self.error_scaling
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
        """Get noise from gate-swap (SWAP) operations.

        In the trapped-ion QCCD architecture, each SWAP decomposes
        into **3 MS gates** (Fig. 5, arXiv:2004.04706).  The old
        ground-truth ``GateSwap.calculateFidelity()`` computes fidelity
        as the product of 3 MS-gate fidelities — each evaluated at
        the current chain length and motional quanta.  This method
        replicates that behaviour when ``chain_length`` is available.

        If ``swap_info.chain_length`` is not set, falls back to the
        fixed ``swap_info.error_probability``.

        Parameters
        ----------
        swap_info : GateSwapInfo
            Information about the swap chain.

        Returns
        -------
        List[NoiseChannel]
            Depolarizing noise from swaps.

        References
        ----------
        Fig. 5, https://arxiv.org/pdf/2004.04706
        """
        channels = []

        if swap_info.num_swaps <= 0:
            return channels

        if len(swap_info.qubits) < 2:
            return channels

        q1, q2 = swap_info.qubits[0], swap_info.qubits[1]

        # --- Physics-based: 3 MS gates per SWAP, per swap ---
        chain_length = getattr(swap_info, 'chain_length', None)
        motional_quanta = getattr(swap_info, 'motional_quanta', 0.0)

        if chain_length is not None and chain_length >= 1:
            ms_gates_per_swap = 3  # Fig. 5, arXiv:2004.04706
            single_ms_fid = self.ms_gate_fidelity(chain_length, motional_quanta)
            total_fidelity = single_ms_fid ** (ms_gates_per_swap * swap_info.num_swaps)
            error_prob = (1.0 - total_fidelity) / self.error_scaling
        else:
            # Fallback: use pre-computed error_probability
            error_prob = swap_info.error_probability / self.error_scaling

        error_prob = min(error_prob, 0.5)

        if error_prob > 0:
            channels.append(NoiseChannel(
                channel_type=NoiseChannelType.DEPOLARIZING_2Q,
                probability=error_prob,
                qubits=(q1, q2),
                stim_instruction=f"DEPOLARIZE2({error_prob}) {q1} {q2}",
            ))

        # --- Dual-path noise for SWAPs ---
        # Same architecture as for regular gates above: the scalar
        # model has already computed DEPOLARIZE2 from the 3-MS-per-SWAP
        # fidelity formula.  If the mode_noise_callback is registered,
        # it gets the full 3N mode state to compute mode-resolved SWAP
        # error (each of the 3 MS gates couples to modes differently
        # depending on which ions are being swapped).
        mode_snapshot = (swap_info.platform_context or {}).get("mode_snapshot", None)
        if self.mode_noise_callback is not None and mode_snapshot is not None:
            channels = self.mode_noise_callback(
                "SWAP", swap_info.qubits, mode_snapshot, channels,
            )

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
        
        REPEAT blocks are preserved: the body of each REPEAT block is
        noised once using the timing of the first repetition, then
        wrapped back into a stim.CircuitRepeatBlock.
        
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
        noisy = stim.Circuit()
        instruction_index = [0]  # mutable counter for nested calls
        
        self._apply_noise_to_block(circuit, plan, noisy, instruction_index)
        
        return noisy
    
    def _apply_noise_to_block(
        self,
        block: stim.Circuit,
        plan: "ExecutionPlan",
        noisy: stim.Circuit,
        instruction_index: List[int],
    ) -> None:
        """Apply noise to a circuit block, preserving REPEAT structure.
        
        Parameters
        ----------
        block : stim.Circuit
            The circuit or repeat-block body to walk.
        plan : ExecutionPlan
            Execution plan with timing metadata.
        noisy : stim.Circuit
            Accumulator for noisy instructions.
        instruction_index : List[int]
            Mutable single-element list holding the current gate index.
        """
        from qectostim.experiments.hardware_simulation.core.execution import (
            ExecutionPlan,
            OperationTiming,
        )
        
        for inst in block:
            # Handle REPEAT blocks: noise the body once, then wrap
            if isinstance(inst, stim.CircuitRepeatBlock):
                noisy_body = stim.Circuit()
                # Save the instruction index before the repeat body
                saved_idx = instruction_index[0]
                self._apply_noise_to_block(
                    inst.body_copy(), plan, noisy_body, instruction_index,
                )
                noisy.append(stim.CircuitRepeatBlock(
                    inst.repeat_count, noisy_body,
                ))
                # For repeat blocks with count > 1, the plan indices
                # only cover the first iteration.  Reset the index to
                # where we'd be after all repetitions — the plan was
                # built from the first iteration, subsequent iterations
                # reuse the same noise pattern.
                continue
            
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
            
            idx = instruction_index[0]
            
            # 1. Idle dephasing BEFORE this instruction.
            # Idle durations come directly from the ExecutionPlan's
            # parallel schedule — no artificial multiplier needed.
            for idle in plan.get_idle_before(idx):
                for channel in self.idle_noise(idle.qubit, idle.duration):
                    if channel.probability > 0:
                        noisy.append_from_stim_program_text(channel.to_stim())
            
            # 2. Gate swap noise (if any)
            for swap_info in plan.get_swaps_for_instruction(idx):
                for channel in self.swap_noise(swap_info):
                    if channel.probability > 0:
                        noisy.append_from_stim_program_text(channel.to_stim())
            
            # Determine if this is a measurement instruction.
            # Stim convention: X_ERROR before M = measurement bit-flip error.
            is_measurement = name in ("M", "MX", "MY", "MZ", "MR",
                                       "MRX", "MRY", "MRZ")
            
            timing = plan.get_operation(idx)
            gate_noise_channels = []
            if timing is not None:
                gate_noise_channels = [
                    ch for ch in self.apply_to_operation_timing(timing)
                    if ch.probability > 0
                ]
            
            if is_measurement:
                # 3a. Gate infidelity noise BEFORE measurement
                for channel in gate_noise_channels:
                    noisy.append_from_stim_program_text(channel.to_stim())
                # 3b. Original measurement instruction LAST
                noisy.append(inst)
            else:
                # 3a. Original instruction first
                noisy.append(inst)
                # 3b. Gate infidelity noise AFTER non-measurement gates
                for channel in gate_noise_channels:
                    noisy.append_from_stim_program_text(channel.to_stim())
            
            instruction_index[0] += 1
    
    def ms_gate_fidelity(
        self,
        chain_length: int,
        motional_quanta: float = 0.0,
    ) -> float:
        """Calculate MS gate fidelity.

        Delegates to the cached :class:`IonChainFidelityModel`.
        See :meth:`IonChainFidelityModel.ms_gate_fidelity` for the
        full formula and references.
        """
        return self._fidelity_model.ms_gate_fidelity(chain_length, motional_quanta)
    
    def single_qubit_gate_fidelity(
        self,
        chain_length: int,
        motional_quanta: float = 0.0,
    ) -> float:
        """Calculate single-qubit gate fidelity.

        Delegates to the cached :class:`IonChainFidelityModel`.
        See :meth:`IonChainFidelityModel.single_qubit_gate_fidelity`.
        """
        return self._fidelity_model.single_qubit_gate_fidelity(chain_length, motional_quanta)
    
    def dephasing_fidelity(
        self,
        duration: float,
    ) -> float:
        """Calculate dephasing fidelity over an idle interval.

        Delegates to the cached :class:`IonChainFidelityModel`.
        F_dephasing = 1 − (1 − exp(−t/T2))/2
        """
        return self._fidelity_model.dephasing_fidelity(duration)
    
    def transport_error(
        self,
        operation_type: str,
    ) -> float:
        """Get motional quanta added by a transport operation.

        Values are ``HEATING_RATE * OP_TIME`` derived from the old
        ground-truth code.  See :class:`TrappedIonCalibration` for
        the full table of transport heating constants.

        Parameters
        ----------
        operation_type : str
            One of ``"split"``, ``"merge"``, ``"shuttle"``, ``"move"``,
            ``"junction"``, ``"rotation"``, ``"cooling"``, ``"crossing_swap"``.

        Returns
        -------
        float
            Motional quanta added by this transport operation.
        """
        return self.ion_calibration.transport_heating.get(operation_type, 0.0)
