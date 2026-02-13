#!/usr/bin/env python3
"""Check CNOT-HTEL shortest path more carefully - is it really 1?"""
import stim
from qectostim.codes import SteaneCode713
from qectostim.experiments.ft_gadget_experiment import FaultTolerantGadgetExperiment
from qectostim.gadgets.teleportation_h_gadgets import CNOTHTeleportGadget
from qectostim.noise.models import StimStyleDepolarizingNoise

code = SteaneCode713()
noise = StimStyleDepolarizingNoise(p=1e-8)

gadget = CNOTHTeleportGadget(input_state="0")
exp = FaultTolerantGadgetExperiment(
    codes=[code], gadget=gadget, noise_model=noise,
    num_rounds_before=3, num_rounds_after=3,
)
circ = exp.to_stim()
dem = circ.detector_error_model(decompose_errors=False)

# Find shortest graphlike error with L flip
shortest_errors = []
for error in dem:
    if error.type != "error":
        continue
    targets = error.targets_copy()
    dets = [t for t in targets if t.is_relative_detector_id()]
    obs = [t for t in targets if t.is_logical_observable_id()]
    
    if len(obs) > 0 and len(dets) <= 2:
        prob = error.args_copy()[0]
        det_ids = [t.val for t in dets]
        obs_ids = [t.val for t in obs]
        shortest_errors.append((len(dets), prob, det_ids, obs_ids))

shortest_errors.sort(key=lambda x: (x[0], -x[1]))
print(f"CNOT-HTEL 0: {circ.num_detectors} det, errors with L flip and <=2 det symptoms:")
for w, p, d, o in shortest_errors[:20]:
    print(f"  weight={w}, prob={p:.3e}, dets={d}, obs={o}")

# Also check with decompose_errors=True
dem2 = circ.detector_error_model(decompose_errors=True)
shortest2 = []
for error in dem2:
    if error.type != "error":
        continue
    targets = error.targets_copy()
    dets = [t for t in targets if t.is_relative_detector_id()]
    obs = [t for t in targets if t.is_logical_observable_id()]
    
    if len(obs) > 0 and len(dets) <= 2:
        prob = error.args_copy()[0]
        det_ids = [t.val for t in dets]
        obs_ids = [t.val for t in obs]
        shortest2.append((len(dets), prob, det_ids, obs_ids))

shortest2.sort(key=lambda x: (x[0], -x[1]))
print(f"\nWith decompose_errors=True:")
for w, p, d, o in shortest2[:20]:
    print(f"  weight={w}, prob={p:.3e}, dets={d}, obs={o}")
