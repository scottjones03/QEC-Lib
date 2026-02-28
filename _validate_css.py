"""Validate CSS surgery correctness after bridge fix."""
import sys; sys.path.insert(0, 'src')
from qectostim.codes.surface.rotated_surface import RotatedSurfaceCode
from qectostim.gadgets.css_surgery_cnot import CSSSurgeryCNOTGadget
from qectostim.experiments.ft_gadget_experiment import FaultTolerantGadgetExperiment
import numpy as np

for d in [2, 3]:
    code = RotatedSurfaceCode(distance=d)
    gadget = CSSSurgeryCNOTGadget()
    ft = FaultTolerantGadgetExperiment(codes=[code], gadget=gadget, noise_model=None, num_rounds_before=d, num_rounds_after=d)
    circ = ft.to_stim()
    print(f'd={d}: {circ.num_qubits} qubits, {len(circ)} instructions')
    sampler = circ.compile_detector_sampler()
    dets, obs = sampler.sample(1000, separate_observables=True)
    false_det = int(dets.sum())
    false_obs = int(obs.sum())
    print(f'  False detectors: {false_det}, False obs: {false_obs}')
    assert false_det == 0 and false_obs == 0, f'FAIL at d={d}'
    print(f'  PASS')
print('All CSS surgery correctness checks passed')
