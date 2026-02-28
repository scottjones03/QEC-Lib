"""Reproduce the NoFeasibleLayoutError bug with CSS Surgery gadget."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from qectostim.codes.surface.rotated_surface import RotatedSurfaceCode
from qectostim.gadgets.css_surgery_cnot import CSSSurgeryCNOTGadget
from qectostim.experiments.ft_gadget_experiment import FaultTolerantGadgetExperiment

d = 2; k = 2
gadget_cnot = CSSSurgeryCNOTGadget()
code_d2 = RotatedSurfaceCode(distance=d)
ft_exp = FaultTolerantGadgetExperiment(
    codes=[code_d2], gadget=gadget_cnot, noise_model=None,
    num_rounds_before=d, num_rounds_after=d,
)
ideal_gadget = ft_exp.to_stim()
qec_meta = ft_exp.qec_metadata
qubit_alloc = ft_exp._unified_allocation

print(f'Ideal circuit: {len(ideal_gadget)} instructions, {ideal_gadget.num_qubits} qubits')
print(f'Blocks: {[ba.block_name for ba in qec_meta.block_allocations]}')
print(f'Phases: {len(qec_meta.phases)}')
for i, ph in enumerate(qec_meta.phases):
    print(f'  [{i}] {ph.phase_type!s:10s}  blocks={ph.active_blocks}')

import traceback, logging
logging.basicConfig(level=logging.INFO, format='%(name)s %(levelname)s: %(message)s')

try:
    from qectostim.experiments.hardware_simulation.trapped_ion.demo.run import compile_gadget_for_animation
    result = compile_gadget_for_animation(
        ideal_gadget,
        qec_metadata=qec_meta,
        gadget=gadget_cnot,
        qubit_allocation=qubit_alloc,
        trap_capacity=k,
        lookahead=1,
        subgridsize=(12, 12, 0),
        base_pmax_in=1,
        show_progress=False,
    )
    print('SUCCESS')
except Exception as e:
    traceback.print_exc()
