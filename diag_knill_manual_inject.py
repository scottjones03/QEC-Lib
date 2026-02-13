#!/usr/bin/env python3
"""Use Stim's exact error propagation to check what happens with XX on Q29,Q36 at tick 117."""
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

# Get the DEM
dem = circ.detector_error_model(decompose_errors=False)

# Print all undetectable errors with full details
for error in dem:
    if error.type != "error":
        continue
    targets = error.targets_copy()
    dets = [t for t in targets if t.is_relative_detector_id()]
    obs = [t for t in targets if t.is_logical_observable_id()]
    if len(dets) == 0 and len(obs) > 0:
        print(f"Undetectable: prob={error.args_copy()[0]:.6e}")
        print(f"  Obs: {[t.val for t in obs]}")
        print(f"  Full: {error}")
        
        # Now trace it
        explained = circ.explain_detector_error_model_errors(
            dem_filter=stim.DetectorErrorModel(str(error)),
            reduce_to_one_representative_error=False
        )
        print(f"  {len(explained)} representative error(s)")
        for exp_err in explained:
            for loc in exp_err.circuit_error_locations:
                print(f"  Location: tick={loc.tick_offset}")
                print(f"    Pauli: {loc.flipped_pauli_product}")
                print(f"    Gate: {loc.instruction_targets.gate}")

# Let's also verify: manually insert X errors and see what detectors fire
print("\n=== Manual verification: insert X29 X36 error ===")

# Strip noise and detectors to get bare circuit, then add back detectors
bare = stim.Circuit()
for inst in circ.flattened():
    if "DEPOLARIZE" not in inst.name and "NOISE" not in inst.name:
        bare.append(inst)

# Now sample with and without the X error
# Use the tableau simulator
sim = stim.TableauSimulator()
sim.do(bare)  # This runs the whole circuit

# Actually, let's use a different approach: use the detector sampler
# with a specific error injected

# Manual approach: create a circuit with ONLY the XX error at the right spot
noisy = stim.Circuit()
tick = 0
injected = False
for inst in circ.flattened():
    # Skip all existing noise
    if "DEPOLARIZE" in inst.name or "NOISE" in inst.name:
        continue
    
    noisy.append(inst)
    
    if inst.name == "TICK":
        tick += 1
        # After reaching tick 117 and the CX, inject the error
        if tick == 118 and not injected:
            # X error on Q29 and Q36 (after the CX at tick 117)
            noisy.append(stim.CircuitInstruction("X_ERROR", [29, 36], [1.0]))
            injected = True

print(f"Injected error: {injected}")
print(f"Noisy circuit detectors: {noisy.num_detectors}")
print(f"Noisy circuit observables: {noisy.num_observables}")

# Sample detectors
det_sampler = noisy.compile_detector_sampler()
samples = det_sampler.sample(shots=1, append_observables=True)
det_part = samples[0, :noisy.num_detectors]
obs_part = samples[0, noisy.num_detectors:]

fired_dets = [i for i in range(len(det_part)) if det_part[i]]
fired_obs = [i for i in range(len(obs_part)) if obs_part[i]]

print(f"  Fired detectors: {fired_dets}")
print(f"  Fired observables: {fired_obs}")

# Also try with error at a different position - between tick 117 CX and DEPOLARIZE
# Actually the DEPOLARIZE2 is AFTER the CX, so the error is after the gate
# But I injected it after TICK (which increments to 118)
# Let me try injecting right after the specific CX at tick 117

noisy2 = stim.Circuit()
tick = 0
injected2 = False
for inst in circ.flattened():
    if "DEPOLARIZE" in inst.name or "NOISE" in inst.name:
        continue
    
    noisy2.append(inst)
    
    if inst.name == "CX" and tick == 117 and not injected2:
        targets = inst.targets_copy()
        qs = [t.value for t in targets]
        pairs = [(qs[i], qs[i+1]) for i in range(0, len(qs), 2)]
        if (29, 36) in pairs:
            noisy2.append(stim.CircuitInstruction("X_ERROR", [29, 36], [1.0]))
            injected2 = True
            print(f"Injected right after CX at tick {tick}")
    
    if inst.name == "TICK":
        tick += 1

print(f"Injected2: {injected2}")
det_sampler2 = noisy2.compile_detector_sampler()
samples2 = det_sampler2.sample(shots=1, append_observables=True)
det_part2 = samples2[0, :noisy2.num_detectors]
obs_part2 = samples2[0, noisy2.num_detectors:]

fired_dets2 = [i for i in range(len(det_part2)) if det_part2[i]]
fired_obs2 = [i for i in range(len(obs_part2)) if obs_part2[i]]

print(f"  Fired detectors: {fired_dets2}")
print(f"  Fired observables: {fired_obs2}")

print("\nDone.")
