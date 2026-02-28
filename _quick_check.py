#!/usr/bin/env python3
"""Quick check: ms_pair_count vs actual CX pairs."""
import sys; sys.path.insert(0, 'src')
from qectostim.codes.surface.rotated_surface import RotatedSurfaceCode
from qectostim.gadgets.css_surgery_cnot import CSSSurgeryCNOTGadget
from qectostim.experiments.ft_gadget_experiment import FaultTolerantGadgetExperiment
d=2; k=2
gadget=CSSSurgeryCNOTGadget()
code=RotatedSurfaceCode(distance=d)
ft=FaultTolerantGadgetExperiment(codes=[code],gadget=gadget,noise_model=None,num_rounds_before=d,num_rounds_after=d)
ideal=ft.to_stim()
qm=ft.qec_metadata
total=0
for i,ph in enumerate(qm.phases):
    c=getattr(ph,'ms_pair_count',0); total+=c
    print(f'phase {i}: {ph.phase_type:30s} ms={c:3d} active={ph.active_blocks}')
print(f'Total ms_pair_count: {total}')
cx=0
for instr in ideal.flattened():
    if instr.name in ('CX','CNOT','ZCX'): cx+=len(instr.targets_copy())//2
print(f'Actual CX pairs: {cx}')
print(f'Gap: {cx - total} CX pairs unaccounted')
