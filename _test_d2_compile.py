"""Quick test: d=2 surface code compile on WISE with SAT routing."""
from qectostim.codes.surface.rotated_surface import RotatedSurfaceCode
code = RotatedSurfaceCode(2)
print("d=2:", code.n, code.k, code.d)
circ = code.memory_circuit(rounds=1)
print("qubits:", circ.num_qubits)
print("ops:", len(list(circ.flattened())))

from qectostim.experiments.hardware_simulation.trapped_ion import (
    WISEArchitecture, WISECompiler, WISERoutingConfig,
)

nq = circ.num_qubits
rows = 2
k = 2
m = max(1, (nq + rows * k - 1) // (rows * k))
print(f"nq={nq}, rows={rows}, k={k}, m={m}")
arch = WISEArchitecture(col_groups=m, rows=rows, ions_per_segment=k)
print(f"arch: {arch.name}, total_qubits={arch.num_qubits}")
config = WISERoutingConfig(timeout_seconds=30)
compiler = WISECompiler(arch, routing_config=config)
compiled = compiler.compile(circ)
print("Compiled successfully!")
print(f"Operations: {len(compiled.routed.operations)}")
