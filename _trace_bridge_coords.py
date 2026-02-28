#!/usr/bin/env python3
"""Trace exact qubit coordinates for CSS Surgery CNOT d=2."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from qectostim.codes.surface.rotated_surface import RotatedSurfaceCode
from qectostim.gadgets.css_surgery_cnot import CSSSurgeryCNOTGadget

code = RotatedSurfaceCode(distance=2)
print(f"=== RotatedSurfaceCode d=2 ===")
print(f"n = {code.n} data qubits")
print(f"hx shape = {code._hx.shape}  -->  {code._hx.shape[0]} X-stabilisers")
print(f"hz shape = {code._hz.shape}  -->  {code._hz.shape[0]} Z-stabilisers")

md = code.metadata
data_coords = list(md["data_coords"])
print(f"\nData coords (sorted y,x): {data_coords}")

x_stab_coords = code.get_x_stabilizer_coords()
z_stab_coords = code.get_z_stabilizer_coords()
print(f"X-stab coords: {x_stab_coords}")
print(f"Z-stab coords: {z_stab_coords}")

print(f"\nTotal qubits per block: {code.n} data + {len(x_stab_coords)} X-anc + {len(z_stab_coords)} Z-anc = {code.n + len(x_stab_coords) + len(z_stab_coords)}")

gadget = CSSSurgeryCNOTGadget()
layout = gadget.compute_layout([code])

print(f"\n=== GadgetLayout ===")
print(f"Total qubits: {layout.total_qubits}")
print(f"Blocks: {list(layout.blocks.keys())}")

for bname, binfo in layout.blocks.items():
    print(f"\n--- {bname} ---")
    print(f"  offset: {binfo.offset}")
    print(f"  data_range: {list(binfo.data_qubit_range)}")
    print(f"  x_anc_range: {list(binfo.x_ancilla_range)}")
    print(f"  z_anc_range: {list(binfo.z_ancilla_range)}")

print(f"\n=== Bridge Ancillas ===")
for ba in layout.bridge_ancillas:
    print(f"  global_idx={ba.global_idx}, coord={ba.coord}, purpose={ba.purpose}, connected_blocks={ba.connected_blocks}")

print(f"\n=== All Qubit Coordinates (QubitIndexMap) ===")
for gi in sorted(layout.qubit_map.global_coords.keys()):
    coord = layout.qubit_map.global_coords[gi]
    block_info = layout.qubit_map.global_to_block.get(gi, ("?", "?", "?"))
    print(f"  q{gi}: coord={coord}  block_info={block_info}")

print(f"\n=== ZZ Merge Info ===")
zz_info = gadget._zz_merge_info
print(f"  seam_type={zz_info.seam_type}, grown_type={zz_info.grown_type}")
print(f"  num seam stabs: {len(zz_info.seam_stabs)}")
for i, s in enumerate(zz_info.seam_stabs):
    print(f"  seam[{i}]: lattice_pos={s.lattice_position}, type={s.stab_type}, global_anc={s.global_ancilla_idx}, weight={s.weight}")
    print(f"           support={s.support_globals}")
    for ph, cx in enumerate(s.cx_per_phase):
        if cx:
            print(f"           phase[{ph}]: {cx}")
print(f"  num grown stabs: {len(zz_info.grown_stabs)}")
for i, g in enumerate(zz_info.grown_stabs):
    print(f"  grown[{i}]: lattice_pos={g.lattice_position}, type={g.stab_type}, existing_anc={g.existing_ancilla_global}, block={g.belongs_to_block}")
    print(f"             orig_weight={g.original_weight}, new_weight={g.new_weight}")
    for ph, cx in enumerate(g.new_cx_per_phase):
        if cx:
            print(f"             phase[{ph}]: {cx}")

print(f"\n=== XX Merge Info ===")
xx_info = gadget._xx_merge_info
print(f"  seam_type={xx_info.seam_type}, grown_type={xx_info.grown_type}")
print(f"  num seam stabs: {len(xx_info.seam_stabs)}")
for i, s in enumerate(xx_info.seam_stabs):
    print(f"  seam[{i}]: lattice_pos={s.lattice_position}, type={s.stab_type}, global_anc={s.global_ancilla_idx}, weight={s.weight}")
    print(f"           support={s.support_globals}")
    for ph, cx in enumerate(s.cx_per_phase):
        if cx:
            print(f"           phase[{ph}]: {cx}")
print(f"  num grown stabs: {len(xx_info.grown_stabs)}")
for i, g in enumerate(xx_info.grown_stabs):
    print(f"  grown[{i}]: lattice_pos={g.lattice_position}, type={g.stab_type}, existing_anc={g.existing_ancilla_global}, block={g.belongs_to_block}")
    print(f"             orig_weight={g.original_weight}, new_weight={g.new_weight}")
    for ph, cx in enumerate(g.new_cx_per_phase):
        if cx:
            print(f"             phase[{ph}]: {cx}")

import stim
circ = stim.Circuit()
from qectostim.gadgets.layout import get_code_coords, pad_coord_to_dim
for bname, binfo in layout.blocks.items():
    code_b = binfo.code
    offset = binfo.offset
    data_c, x_c, z_c = get_code_coords(code_b)
    for i, gidx in enumerate(binfo.data_qubit_range):
        if i < len(data_c):
            lc = pad_coord_to_dim(data_c[i], 2)
            gc = (lc[0] + offset[0], lc[1] + offset[1])
        else:
            gc = offset
        circ.append("QUBIT_COORDS", [gidx], list(gc))
    for i, gidx in enumerate(binfo.x_ancilla_range):
        if i < len(x_c):
            lc = pad_coord_to_dim(x_c[i], 2)
            gc = (lc[0] + offset[0], lc[1] + offset[1])
        else:
            gc = offset
        circ.append("QUBIT_COORDS", [gidx], list(gc))
    for i, gidx in enumerate(binfo.z_ancilla_range):
        if i < len(z_c):
            lc = pad_coord_to_dim(z_c[i], 2)
            gc = (lc[0] + offset[0], lc[1] + offset[1])
        else:
            gc = offset
        circ.append("QUBIT_COORDS", [gidx], list(gc))

for ba in layout.bridge_ancillas:
    circ.append("QUBIT_COORDS", [ba.global_idx], list(ba.coord))

print(f"\n=== Stim QUBIT_COORDS ===")
for line in str(circ).strip().split("\n"):
    print(f"  {line}")
