#!/usr/bin/env python3
"""Deep investigation: what exact measurement indices are in pre vs post gadget meas?"""
import logging
import stim
from qectostim.codes import SteaneCode713
from qectostim.codes.composite.concatenated import ConcatenatedCSSCode
from qectostim.experiments.ft_gadget_experiment import FaultTolerantGadgetExperiment
from qectostim.gadgets.teleportation_h_gadgets import CNOTHTeleportGadget

logging.basicConfig(level=logging.WARNING)

steane = SteaneCode713()
concat = ConcatenatedCSSCode(steane, steane)
gadget = CNOTHTeleportGadget(input_state="0")

# Monkey-patch emit_crossing_detectors to capture pre/post data
import qectostim.experiments.detector_emission as de
_orig_emit = de.emit_crossing_detectors

captured = {}
def patched_emit(circuit, builders, pre_gadget_meas, crossing_config, 
                 destroyed_blocks, destroyed_block_bases=None):
    # Capture the raw data
    block_builders = {b.block_name: b for b in builders}
    
    post_raw = {}
    for b in builders:
        raw = b.get_last_measurement_indices()
        post_raw[b.block_name] = raw
    
    captured['pre'] = pre_gadget_meas
    captured['post_raw'] = post_raw
    captured['builders'] = builders
    captured['circuit_meas'] = circuit.num_measurements
    
    # Also try compensation on post
    post_comp = {}
    for b in builders:
        raw = b.get_last_measurement_indices()
        comp = de._compensate_hierarchical_meas(b, raw)
        post_comp[b.block_name] = comp
    captured['post_comp'] = post_comp
    
    # Call original
    return _orig_emit(circuit, builders, pre_gadget_meas, crossing_config,
                      destroyed_blocks, destroyed_block_bases)

de.emit_crossing_detectors = patched_emit

exp = FaultTolerantGadgetExperiment(
    codes=[concat], gadget=gadget, noise_model=None,
    num_rounds_before=2, num_rounds_after=2, d_inner=3,
)
circuit = exp.to_stim()

print(f"Circuit: {circuit.num_qubits} qubits, {circuit.num_measurements} meas")
print(f"Detectors: {circuit.num_detectors}")

# Examine the captured data
for block_name in ['data_block', 'ancilla_block']:
    print(f"\n{'='*60}")
    print(f"  Block: {block_name}")
    print(f"{'='*60}")
    
    pre = captured['pre'].get(block_name, {})
    post_raw = captured['post_raw'].get(block_name, {})
    post_comp = captured['post_comp'].get(block_name, {})
    
    for basis in ['x', 'z']:
        pre_list = pre.get(basis, [])
        post_r = post_raw.get(basis, [])
        post_c = post_comp.get(basis, [])
        
        print(f"\n  --- {basis.upper()} stabilizers ---")
        print(f"  pre entries: {len(pre_list)}, post_raw: {len(post_r)}, post_comp: {len(post_c)}")
        
        # Show first few inner entries
        for i in range(min(3, len(pre_list))):
            p = pre_list[i] if i < len(pre_list) else None
            r = post_r[i] if i < len(post_r) else None
            c = post_c[i] if i < len(post_c) else None
            print(f"    [{i}] pre={p}  post_raw={r}  post_comp={c}")
        
        # Show outer entries
        if len(pre_list) > 21:
            print(f"    ...")
            for i in range(21, min(24, len(pre_list))):
                p = pre_list[i] if i < len(pre_list) else None
                r = post_r[i] if i < len(post_r) else None
                c = post_c[i] if i < len(post_c) else None
                print(f"    [{i}] pre={p}  post_raw={r}  post_comp={c}")

# Now test individual crossing detector flows manually
print(f"\n{'='*60}")
print(f"  Manual has_flow tests")
print(f"{'='*60}")

total_meas = circuit.num_measurements
pre_data_z = captured['pre'].get('data_block', {}).get('z', [])
post_data_z = captured['post_raw'].get('data_block', {}).get('z', [])

# Test Z_D formula: pre_Z_data ⊕ post_Z_data
for i in range(min(3, len(pre_data_z))):
    pre_entry = pre_data_z[i]
    post_entry = post_data_z[i]
    
    # Build targets
    targets = []
    if isinstance(pre_entry, list):
        targets.extend(pre_entry)
    elif pre_entry is not None:
        targets.append(pre_entry)
    if isinstance(post_entry, list):
        targets.extend(post_entry)
    elif post_entry is not None:
        targets.append(post_entry)
    
    # Convert to stim flow
    meas_indices = [idx - total_meas for idx in targets]
    flow = stim.Flow(
        input=stim.PauliString(circuit.num_qubits),
        output=stim.PauliString(circuit.num_qubits),
        measurements=meas_indices,
    )
    result = circuit.has_flow(flow, unsigned=True)
    print(f"  Z_D[{i}] raw pre+raw post: has_flow={result} (targets={targets[:4]}...)")

# Test with stripped pre (just inner meas) + raw post for inner entry
print(f"\n  --- Testing inner Z_D[0] with different compensation combos ---")
i = 0
pre_entry = pre_data_z[i]
post_entry = post_data_z[i]

if isinstance(pre_entry, list):
    raw_pre_inner = pre_entry[0]
    comp_pre = pre_entry
else:
    raw_pre_inner = pre_entry
    comp_pre = [pre_entry]

post_comp_entry = captured['post_comp'].get('data_block', {}).get('z', [])[i]
if isinstance(post_comp_entry, list):
    comp_post = post_comp_entry
else:
    comp_post = [post_comp_entry]

combos = [
    ("raw_pre + raw_post", [raw_pre_inner, post_entry]),
    ("comp_pre + raw_post", comp_pre + [post_entry]),
    ("raw_pre + comp_post", [raw_pre_inner] + comp_post),
    ("comp_pre + comp_post", comp_pre + comp_post),
]

for label, targets in combos:
    meas_indices = [idx - total_meas for idx in targets]
    flow = stim.Flow(
        input=stim.PauliString(circuit.num_qubits),
        output=stim.PauliString(circuit.num_qubits),
        measurements=meas_indices,
    )
    result = circuit.has_flow(flow, unsigned=True)
    print(f"    {label}: has_flow={result}  meas={targets}")
