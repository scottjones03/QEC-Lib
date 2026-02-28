"""Enumerate the rotated surface code layout for d=2 and d=3."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
import numpy as np
from qectostim.codes.surface.rotated_surface import RotatedSurfaceCode

for d in [2, 3]:
    code = RotatedSurfaceCode(distance=d)
    meta = code.metadata
    print(f"\n{'='*70}")
    print(f"  RotatedSurfaceCode d={d}")
    print(f"{'='*70}")
    
    data_coords = meta["data_coords"]
    x_stab_coords = meta["x_stab_coords"]
    z_stab_coords = meta["z_stab_coords"]
    
    print(f"\nData qubits ({len(data_coords)}):")
    for i, c in enumerate(data_coords):
        print(f"  idx={i}: ({c[0]}, {c[1]})")
    
    print(f"\nX-stabilizer ancillas ({len(x_stab_coords)}):")
    for i, c in enumerate(x_stab_coords):
        # find neighbors
        nbrs = []
        for dx, dy in [(1,1),(1,-1),(-1,1),(-1,-1)]:
            n = (c[0]+dx, c[1]+dy)
            if n in [tuple(dc) for dc in data_coords]:
                idx = next(j for j,dc in enumerate(data_coords) if tuple(dc)==n)
                nbrs.append((n, idx))
        print(f"  ({c[0]}, {c[1]}): weight-{len(nbrs)}, "
              f"data={[(n[1], n[0]) for n in nbrs]}")
    
    print(f"\nZ-stabilizer ancillas ({len(z_stab_coords)}):")
    for i, c in enumerate(z_stab_coords):
        nbrs = []
        for dx, dy in [(1,1),(1,-1),(-1,1),(-1,-1)]:
            n = (c[0]+dx, c[1]+dy)
            if n in [tuple(dc) for dc in data_coords]:
                idx = next(j for j,dc in enumerate(data_coords) if tuple(dc)==n)
                nbrs.append((n, idx))
        print(f"  ({c[0]}, {c[1]}): weight-{len(nbrs)}, "
              f"data={[(n[1], n[0]) for n in nbrs]}")
    
    # Boundaries
    for edge in ["left", "right", "top", "bottom"]:
        bq = code.get_boundary_qubits(edge, 0)
        bc = code.get_boundary_coords(edge, 0)
        print(f"\nBoundary '{edge}':")
        print(f"  qubit indices: {bq}")
        print(f"  coords: {bc}")
    
    # hx, hz
    print(f"\nhx shape: {code.hx.shape}")
    print(f"hx =\n{code.hx}")
    print(f"\nhz shape: {code.hz.shape}")
    print(f"hz =\n{code.hz}")
    
    # Logical operators
    print(f"\nLogical X support: {meta.get('lx_support')}")
    print(f"Logical X coords: {[data_coords[i] for i in meta.get('lx_support', [])]}")
    print(f"Logical Z support: {meta.get('lz_support')}")
    print(f"Logical Z coords: {[data_coords[i] for i in meta.get('lz_support', [])]}")
    
    # Schedules
    print(f"\nX schedule (ancilla→data offsets): {meta.get('x_schedule')}")
    print(f"Z schedule (data→ancilla offsets): {meta.get('z_schedule')}")
    
    # ASCII art
    print(f"\nASCII lattice (even coords = ancillas, odd = data):")
    max_coord = 2*d
    for y in range(0, max_coord+1):
        row = ""
        for x in range(0, max_coord+1):
            c = (float(x), float(y))
            if c in [tuple(dc) for dc in data_coords]:
                idx = next(j for j,dc in enumerate(data_coords) if tuple(dc)==c)
                row += f"D{idx} "
            elif c in [tuple(sc) for sc in x_stab_coords]:
                row += "X  "
            elif c in [tuple(sc) for sc in z_stab_coords]:
                row += "Z  "
            else:
                row += ".  "
        print(f"  y={y}: {row}")
