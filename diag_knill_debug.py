#!/usr/bin/env python3
"""Debug: why aren't boundary detectors being emitted for KnillEC destroyed blocks?

Add instrumentation to track the actual flow through _emit_boundary_detectors.
"""
import stim
from qectostim.codes import SteaneCode713
from qectostim.gadgets.knill_ec import KnillECGadget
from qectostim.experiments.ft_gadget_experiment import FaultTolerantGadgetExperiment
from qectostim.noise.models import StimStyleDepolarizingNoise

code = SteaneCode713()
gadget = KnillECGadget(input_state="0")

# Check what the boundary config returns
bc = gadget.get_boundary_detector_config()
print(f"Boundary config: {bc.block_configs}")

exp = FaultTolerantGadgetExperiment(
    codes=[code], gadget=gadget, noise_model=None,
    num_rounds_before=3, num_rounds_after=3,
)

# Monkey-patch _emit_boundary_detectors to add logging
original_emit = exp._emit_boundary_detectors

def traced_emit(circuit, builders, surviving_alloc, meas_start, 
                destroyed_blocks, destroyed_block_meas_starts):
    print(f"\n_emit_boundary_detectors called:")
    print(f"  builders: {[b.block_name for b in builders]}")
    print(f"  surviving_alloc keys: {list(surviving_alloc.keys())}")
    print(f"  meas_start: {meas_start}")
    print(f"  destroyed_blocks: {destroyed_blocks}")
    print(f"  destroyed_block_meas_starts: {destroyed_block_meas_starts}")
    
    boundary_config = gadget.get_boundary_detector_config()
    print(f"  boundary_config.block_configs: {boundary_config.block_configs}")
    
    surviving_builders = [b for b in builders if b.block_name not in destroyed_blocks]
    print(f"  surviving_builders: {[b.block_name for b in surviving_builders]}")
    
    destroyed_builders = [b for b in builders if b.block_name in destroyed_block_meas_starts]
    print(f"  destroyed_builders: {[b.block_name for b in destroyed_builders]}")
    
    return original_emit(circuit, builders, surviving_alloc, meas_start,
                        destroyed_blocks, destroyed_block_meas_starts)

exp._emit_boundary_detectors = traced_emit

base = exp.to_stim()

print(f"\nCircuit detectors: {sum(1 for i in base.flattened() if i.name == 'DETECTOR')}")

# Now also check if auto_detectors is overriding things
print(f"\nauto_detectors: {exp.auto_detectors}")
print(f"gadget.use_auto_detectors(): {gadget.use_auto_detectors()}")

# KnillEC returns True for use_auto_detectors!
# This means the manual detectors are being REPLACED by auto-discovered ones!
print("\n⚠️  KnillEC uses auto_detectors! The manual boundary detectors might be")
print("replaced by auto-discovered ones that don't include destroyed block boundaries.")

print("\nDone.")
