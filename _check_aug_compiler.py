"""Quick check of AugGrid compiler + RotatedSurfaceCode capabilities."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from qectostim.experiments.hardware_simulation.trapped_ion.architecture import (
    AugmentedGridArchitecture, WISEArchitecture,
)
from qectostim.experiments.hardware_simulation.trapped_ion.compilers import AugmentedGridCompiler
from qectostim.codes.small.steane_713 import SteaneCode713
from qectostim.codes.surface.rotated_surface import RotatedSurfaceCode

# Check d=2 surface code
code = RotatedSurfaceCode(2)
print(f'Rotated surface d=2: n={code.n}, k={code.k}')
print(f'  x_stabilizers: {len(code.x_stabilizers())}')
print(f'  z_stabilizers: {len(code.z_stabilizers())}')

# AugGrid for 4 data qubits
arch = AugmentedGridArchitecture(rows=2, cols=2, ions_per_trap=3)
print(f'AugGrid 2x2: {arch.num_qubits} qubits, {len(arch.qccd_graph.nodes)} nodes')

# AugGridCompiler
comp = AugmentedGridCompiler(arch)
print(f'Compiler: {type(comp).__name__}')
methods = [m for m in dir(comp) if not m.startswith('_') and callable(getattr(comp, m))]
print(f'  methods: {methods}')

# Check TrappedIonExperiment
from qectostim.experiments.hardware_simulation.trapped_ion.experiment import TrappedIonExperiment
from qectostim.experiments.hardware_simulation.trapped_ion.noise import TrappedIonNoiseModel

# Try building experiment with Steane code (7 qubits) on a bigger grid
arch2 = AugmentedGridArchitecture(rows=2, cols=3, ions_per_trap=4)
print(f'\nAugGrid 2x3 k=4: {arch2.num_qubits} qubits')
steane = SteaneCode713()
comp2 = AugmentedGridCompiler(arch2)

try:
    exp = TrappedIonExperiment(
        code=steane, architecture=arch2,
        compiler=comp2,
        hardware_noise=TrappedIonNoiseModel(), rounds=1,
    )
    ideal = exp.build_ideal_circuit()
    print(f'Ideal circuit: {ideal.num_qubits} qubits, {len(ideal)} instructions')

    native = comp2.decompose_to_native(ideal)
    mapped = comp2.map_qubits(native)
    routed = comp2.route(mapped)
    ops = list(routed)
    print(f'Routed: {len(ops)} ops')
    # Count types
    from collections import Counter
    type_counts = Counter(type(o).__name__ for o in ops)
    print(f'Op types: {dict(type_counts)}')
except Exception as e:
    import traceback
    traceback.print_exc()
    print(f'Error: {e}')
