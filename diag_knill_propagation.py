#!/usr/bin/env python3
"""Trace CNOT schedule precisely around tick 117 to understand error propagation."""
import stim
from qectostim.codes import SteaneCode713
from qectostim.gadgets.knill_ec import KnillECGadget
from qectostim.experiments.ft_gadget_experiment import FaultTolerantGadgetExperiment
from qectostim.noise.models import StimStyleDepolarizingNoise

code = SteaneCode713()
noise = StimStyleDepolarizingNoise(p=1e-8)
gadget = KnillECGadget(input_state="0")
exp = FaultTolerantGadgetExperiment(
    codes=[code], gadget=gadget, noise_model=noise,
    num_rounds_before=3, num_rounds_after=3,
)
circ = exp.to_stim()

# Extract CX schedule around tick 117
print("=== CX schedule for Z syndrome extraction (bell_b) around tick 117 ===")
tick = 0
for inst in circ.flattened():
    if inst.name == "TICK":
        tick += 1
        continue
    if 115 <= tick <= 125 and inst.name in ("CX", "MR", "MRX", "R", "RX", "H"):
        targets = inst.targets_copy()
        qs = [t.value for t in targets if hasattr(t, 'value') and not t.is_measurement_record_target]
        if any(q in range(26, 39) for q in qs):
            # For CX, show as pairs
            if inst.name == "CX":
                pairs = [(qs[i], qs[i+1]) for i in range(0, len(qs), 2)]
                print(f"  tick={tick}: CX pairs: {pairs}")
            else:
                print(f"  tick={tick}: {inst.name} qubits={qs}")

# Now: propagate XX error from tick 117 through the rest of the circuit
# X on Q29 after tick 117 CX Q29→Q36 DEPOLARIZE2
# X on Q36 after tick 117 CX Q29→Q36 DEPOLARIZE2
print("\n=== Error propagation: X on Q29 ===")
print("X on Q29 is on a DATA qubit. Through subsequent CNOTs:")
print("  CX data→Z_anc: X on control stays on control (no propagation)")
print("  CX X_anc→data: X on target stays on target (no propagation)")  
print("  So X on Q29 stays on Q29 until M(Q29) at tick 174")
print("  M(Q29) measures Z basis. X anti-commutes with Z → flips measurement")

print("\n=== Error propagation: X on Q36 ===")
print("X on Q36 is on a Z ancilla qubit.")
# After tick 117 DEPOLARIZE2, what CNOTs act on Q36?
tick = 0
for inst in circ.flattened():
    if inst.name == "TICK":
        tick += 1
        continue
    if tick > 117 and tick <= 125 and inst.name == "CX":
        targets = inst.targets_copy()
        qs = [t.value for t in targets]
        pairs = [(qs[i], qs[i+1]) for i in range(0, len(qs), 2)]
        for ctrl, tgt in pairs:
            if ctrl == 36 or tgt == 36:
                print(f"  tick={tick}: CX Q{ctrl}→Q{tgt}")
                if tgt == 36:
                    print(f"    X on Q36 (target): stays as X on Q36")
                else:
                    print(f"    X on Q36 (control): X_36 → X_36 ⊗ X_{tgt}")

# Check: is Q36 a Z-type ancilla? It should be measured in Z basis.
# CX data→Z_anc means Q36 is the TARGET of the CNOT
# For Z syndrome extraction: CX(data_i, Z_anc) where Z_anc is target
# After CNOT: X_anc → X_anc (target, X stays)
# So X error on Z ancilla just stays there through the round
# It flips the MR measurement (which measures Z basis)

# KEY: But does X on Q36 propagate to Q29 through the tick 118 CNOT?
print("\n=== Critical: tick 118 CX involving Q36 ===")
tick = 0
for inst in circ.flattened():
    if inst.name == "TICK":
        tick += 1
        continue
    if tick == 118 and inst.name == "CX":
        targets = inst.targets_copy()
        qs = [t.value for t in targets]
        pairs = [(qs[i], qs[i+1]) for i in range(0, len(qs), 2)]
        for ctrl, tgt in pairs:
            if ctrl == 36 or tgt == 36:
                print(f"  CX Q{ctrl}→Q{tgt}")
                if tgt == 36:
                    print(f"    Z syndrome: data→Z_anc, X on Z_anc stays")
                    # But also: Z on data propagates to Z on target!
                    # For us: X on Q36 stays X on Q36
                else:
                    # Q36 is control: X on control → X_control ⊗ X_target  
                    print(f"    Q36 is CONTROL: X_Q36 → X_Q36 ⊗ X_Q{tgt}")
                    print(f"    This would spread X error to Q{tgt}!")

# Also check tick 119
tick = 0
for inst in circ.flattened():
    if inst.name == "TICK":
        tick += 1
        continue
    if tick == 119 and inst.name == "CX":
        targets = inst.targets_copy()
        qs = [t.value for t in targets]
        pairs = [(qs[i], qs[i+1]) for i in range(0, len(qs), 2)]
        for ctrl, tgt in pairs:
            if ctrl == 36 or tgt == 36:
                print(f"  tick=119: CX Q{ctrl}→Q{tgt}")
                if tgt == 36:
                    print(f"    Z syndrome: data→Z_anc, X on Z_anc stays")
                else:
                    print(f"    Q36 is CONTROL: X_Q36 → X_Q36 ⊗ X_Q{tgt}")

print("\nDone.")
