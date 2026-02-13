"""Quick verification that Tier 1 / Tier 2 design works correctly."""
import numpy as np

from src.qectostim.experiments.hardware_simulation.trapped_ion.physics import (
    PhysicalConstants, IonSpecies, TrapParameters,
    CalibrationConstants, DEFAULT_CALIBRATION, DEFAULT_TRAP,
    ModeStructure, ModeSnapshot, IonChainFidelityModel, DEFAULT_FIDELITY_MODEL,
)

# 1. Verify trap is None by default
assert DEFAULT_CALIBRATION.trap is None, f"Expected None, got {DEFAULT_CALIBRATION.trap}"
print("✓ DEFAULT_CALIBRATION.trap is None (Tier 1 default)")

# 2. Verify DEFAULT_TRAP still exists for Tier-2 users
assert isinstance(DEFAULT_TRAP, TrapParameters)
print(f"✓ DEFAULT_TRAP exists: species={DEFAULT_TRAP.ion_species.name}")

# 3. Verify Tier-1 fidelity model works without trap
fm = DEFAULT_FIDELITY_MODEL
f = fm.ms_gate_fidelity(chain_length=5, motional_quanta=0.1)
print(f"✓ Tier-1 ms_gate_fidelity(N=5, nbar=0.1) = {f:.6f}")

# 4. Verify mode-resolved fallback to Tier 1
snap = ModeSnapshot(
    n_ions=3,
    mode_frequencies=np.array([1e6] * 9),
    eigenvectors=np.eye(9, 3),
    occupancies=np.array([0.05] * 9),
    scalar_nbar=0.45,
)
f2 = fm.ms_gate_fidelity_mode_resolved(0, 1, snap)
print(f"✓ ms_gate_fidelity_mode_resolved fallback (no trap) = {f2:.6f}")

# 5. Verify Tier-2 with explicit trap
f3 = fm.ms_gate_fidelity_mode_resolved(0, 1, snap, trap=DEFAULT_TRAP)
print(f"✓ ms_gate_fidelity_mode_resolved Tier 2 (with trap) = {f3:.6f}")

# 6. Verify ModeStructure.compute works (Tier 1)
ms = ModeStructure.compute(n_ions=5, axial_freq=1.0e6)
print(f"✓ ModeStructure.compute(N=5): {ms.n_ions} ions, {len(ms.mode_frequencies)} modes")

# 7. Verify dephasing_fidelity (Tier 1)
fd = fm.dephasing_fidelity(duration=1e-3)
print(f"✓ dephasing_fidelity(1ms) = {fd:.6f}")

# 8. Verify transport_phase_error (Tier 1)
tp = fm.transport_phase_error(distance_um=100.0)
print(f"✓ transport_phase_error(100µm) = {tp:.8f}")

# 9. Verify Tier-2 opt-in: create CalibrationConstants with trap
cal_tier2 = CalibrationConstants(trap=DEFAULT_TRAP)
fm2 = IonChainFidelityModel(calibration=cal_tier2)
f4 = fm2.ms_gate_fidelity_mode_resolved(0, 1, snap)
print(f"✓ Tier-2 via CalibrationConstants(trap=...) = {f4:.6f}")

# 10. Verify downstream imports
from src.qectostim.experiments.hardware_simulation.trapped_ion.noise import TrappedIonNoiseModel
print("✓ noise.py imports cleanly")

from src.qectostim.experiments.hardware_simulation.trapped_ion.transport import GatePhysics
print("✓ transport.py imports cleanly")

from src.qectostim.experiments.hardware_simulation.trapped_ion.operations import Split as OpSplit
print("✓ operations.py imports cleanly")

from src.qectostim.experiments.hardware_simulation.trapped_ion.gate_ops import MSGate
print("✓ gate_ops.py imports cleanly")

from src.qectostim.experiments.hardware_simulation.trapped_ion.gates import TrappedIonGateSet
print("✓ gates.py imports cleanly")

from src.qectostim.experiments.hardware_simulation.trapped_ion.execution import TrappedIonExecutionPlanner
print("✓ execution.py imports cleanly")

from src.qectostim.experiments.hardware_simulation.trapped_ion.compilers.base import TrappedIonCompiler
print("✓ compilers/base.py imports cleanly")

print()
print("=" * 50)
print("All Tier 1/2 verification checks passed ✓")
