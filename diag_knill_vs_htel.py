#!/usr/bin/env python3
"""Compare KnillEC and CNOT-HTEL: why does CNOT-HTEL have 0 undetectable errors?

Key insight from prior analysis:
- KnillEC observable: M_Z(bell_a) ⊕ M_Z(bell_b)
- bell_a is measured RIGHT AFTER the Bell CNOT with no syndrome round in between
- Errors on bell_a during the CNOT are undetectable

Question: Does CNOT-HTEL have a similar structure? How does it protect against this?
"""
import stim
from qectostim.codes import SteaneCode713
from qectostim.gadgets.knill_ec import KnillECGadget
from qectostim.gadgets.teleportation_h_gadgets import CNOTHTeleportGadget
from qectostim.experiments.ft_gadget_experiment import FaultTolerantGadgetExperiment
from qectostim.noise.models import StimStyleDepolarizingNoise

code = SteaneCode713()

# Focus on understanding the detector coverage at the measurement boundary
for label, gadget_cls, kwargs in [
    ("KnillEC", KnillECGadget, {"input_state": "0"}),
    ("CNOT-HTEL", CNOTHTeleportGadget, {"input_state": "0"}),
]:
    gadget = gadget_cls(**kwargs)
    exp = FaultTolerantGadgetExperiment(
        codes=[code], gadget=gadget, noise_model=None,
        num_rounds_before=3, num_rounds_after=3,
    )
    base = exp.to_stim()
    
    print(f"\n{'='*70}")
    print(f"{label}")
    print(f"{'='*70}")
    
    flat = list(base.flattened())
    
    # Find the Bell/teleport measurement layer
    # For KnillEC: MX on data_block + M on bell_a
    # For CNOT-HTEL: similar destructive measurement pattern
    
    # Print ALL instructions in the last 50 ticks
    tick_count = 0
    max_tick = 0
    for instr in flat:
        if instr.name == "TICK":
            tick_count += 1
    max_tick = tick_count
    
    print(f"Total ticks: {max_tick}")
    
    tick_count = 0
    print(f"\n--- Last 20 ticks (ticks {max_tick-20} to {max_tick}) ---")
    for i, instr in enumerate(flat):
        if instr.name == "TICK":
            tick_count += 1
        if tick_count >= max_tick - 20:
            tgts = instr.targets_copy()
            gate_args = instr.gate_args_copy()
            tgt_str = ', '.join(str(t.value) if hasattr(t, 'value') else str(t) for t in tgts)
            if instr.name == "DETECTOR":
                # Show detector coords and rec targets
                recs = [str(t) for t in tgts]
                coords = [f"{a:.1f}" for a in gate_args]
                print(f"  [{i}] t={tick_count} DETECTOR({','.join(coords)}) {recs}")
            elif instr.name == "OBSERVABLE_INCLUDE":
                recs = [str(t) for t in tgts]
                print(f"  [{i}] t={tick_count} OBSERVABLE_INCLUDE({int(gate_args[0])}) {recs}")
            elif instr.name in ("M", "MX", "MR", "MRX"):
                qubits = [t.value for t in tgts]
                print(f"  [{i}] t={tick_count} {instr.name} qubits={qubits}")
            elif instr.name == "TICK":
                print(f"  [{i}] t={tick_count} TICK")
            elif instr.name in ("CX", "CNOT"):
                pairs = [(tgts[j].value, tgts[j+1].value) for j in range(0, len(tgts), 2)]
                print(f"  [{i}] t={tick_count} CX pairs={pairs}")
            elif instr.name in ("H", "S", "R", "RX"):
                qubits = [t.value for t in tgts]
                print(f"  [{i}] t={tick_count} {instr.name} qubits={qubits}")
            elif instr.name in ("QUBIT_COORDS", "SHIFT_COORDS"):
                pass  # skip noise coordinates
            elif instr.name.startswith("DEPOLARIZE"):
                pass  # skip (no noise in base circuit)
            else:
                print(f"  [{i}] t={tick_count} {instr.name} {tgt_str}")

    # Count how many detectors check the last syndrome round vs final measurement
    print(f"\n--- Detector analysis ---")
    total_meas = 0
    det_count = 0
    for instr in flat:
        if instr.name in ("M", "MX", "MR", "MRX", "MY"):
            total_meas += len(instr.targets_copy())
        elif instr.name == "DETECTOR":
            det_count += 1
    
    print(f"Total measurements: {total_meas}")
    print(f"Total detectors: {det_count}")

print("\nDone.")
