#!/usr/bin/env python3
"""Research script: analyze rotated surface code structure for lattice surgery."""
import numpy as np
from qectostim.codes.surface.rotated_surface import RotatedSurfaceCode

for d in [3, 5]:
    code = RotatedSurfaceCode(d)
    print(f"=== d={d} ===")
    print(f"n = {code.n}, hx shape: {code.hx.shape}, hz shape: {code.hz.shape}")
    
    dc = code.qubit_coords()
    
    bottom = code.get_boundary_qubits("bottom", 0)
    top = code.get_boundary_qubits("top", 0)
    left = code.get_boundary_qubits("left", 0)
    right = code.get_boundary_qubits("right", 0)
    
    print(f"Boundary bottom ({len(bottom)} qubits): indices={bottom}, coords={[dc[i] for i in bottom]}")
    print(f"Boundary top ({len(top)} qubits): indices={top}, coords={[dc[i] for i in top]}")
    print(f"Boundary left ({len(left)} qubits): indices={left}, coords={[dc[i] for i in left]}")
    print(f"Boundary right ({len(right)} qubits): indices={right}, coords={[dc[i] for i in right]}")
    
    print(f"\nZ stab count: {code.hz.shape[0]}, X stab count: {code.hx.shape[0]}")
    
    # Which Z-stabs touch the bottom boundary?
    print("\nZ-stabs touching bottom boundary qubits:")
    for i in range(code.hz.shape[0]):
        support = set(np.where(code.hz[i])[0])
        overlap = support & set(bottom)
        if overlap:
            print(f"  Z-stab {i}: support={sorted(support)}, overlap with bottom={sorted(overlap)}")
    
    print("\nZ-stabs touching top boundary qubits:")
    for i in range(code.hz.shape[0]):
        support = set(np.where(code.hz[i])[0])
        overlap = support & set(top)
        if overlap:
            print(f"  Z-stab {i}: support={sorted(support)}, overlap with top={sorted(overlap)}")
    
    print("\nX-stabs touching left boundary qubits:")
    for i in range(code.hx.shape[0]):
        support = set(np.where(code.hx[i])[0])
        overlap = support & set(left)
        if overlap:
            print(f"  X-stab {i}: support={sorted(support)}, overlap with left={sorted(overlap)}")
    
    print("\nX-stabs touching right boundary qubits:")
    for i in range(code.hx.shape[0]):
        support = set(np.where(code.hx[i])[0])
        overlap = support & set(right)
        if overlap:
            print(f"  X-stab {i}: support={sorted(support)}, overlap with right={sorted(overlap)}")
    
    # Key question: how many bridge ancillas should we have for ZZ merge (bottom<->top)?
    # In true lattice surgery, we need to measure NEW boundary stabilizers that span
    # the gap. For ZZ merge between rough boundaries:
    # The new stabilizers are Z-type, each involving 2 data qubits (one from each patch)
    # The number of new boundary stabilizers = d-1 for a d-distance code
    print(f"\nFor ZZ merge (bottom-top): should need {d-1} bridge stabilizers")
    print(f"  Bottom has {len(bottom)} data qubits, top has {len(top)} data qubits")
    
    print(f"\nFor XX merge (right-left): should need {d-1} bridge stabilizers")
    print(f"  Right has {len(right)} data qubits, left has {len(left)} data qubits")
    
    # Show which boundary stabilizers are weight-2 (truncated)
    print("\nWeight-2 Z-stabilizers (boundary):")
    for i in range(code.hz.shape[0]):
        support = list(np.where(code.hz[i])[0])
        if len(support) == 2:
            coords = [dc[j] for j in support]
            print(f"  Z-stab {i}: qubits {support}, coords {coords}")
    
    print("\nWeight-2 X-stabilizers (boundary):")
    for i in range(code.hx.shape[0]):
        support = list(np.where(code.hx[i])[0])
        if len(support) == 2:
            coords = [dc[j] for j in support]
            print(f"  X-stab {i}: qubits {support}, coords {coords}")
    
    print()
