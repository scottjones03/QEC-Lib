"""Diagnostic 2: CSS Surgery compiler internals (no SAT routing)."""
import sys, os
os.environ["WISE_INPROCESS_LIMIT"] = "999999999"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from collections import Counter, defaultdict
from qectostim.codes.surface.rotated_surface import RotatedSurfaceCode
from qectostim.experiments.ft_gadget_experiment import FaultTolerantGadgetExperiment
from qectostim.gadgets.css_surgery_cnot import CSSSurgeryCNOTGadget
from qectostim.experiments.hardware_simulation.trapped_ion.utils import WISEArchitecture, TrappedIonCompiler
from qectostim.experiments.hardware_simulation.trapped_ion.utils.qccd_nodes import QCCDWiseArch

d=2; k=2
code = RotatedSurfaceCode(distance=d)
gadget = CSSSurgeryCNOTGadget()
ft = FaultTolerantGadgetExperiment(codes=[code], gadget=gadget, noise_model=None, num_rounds_before=d, num_rounds_after=d)
ideal = ft.to_stim()
meta = ft.qec_metadata
alloc = ft._unified_allocation

# Count CX in stim
cx_count = 0
for inst in ideal.flattened():
    if inst.name in ('CX','CNOT','ZCX','CZ','ZCZ'):
        cx_count += len(inst.targets_copy())//2
print(f'Stim CX pairs: {cx_count}')

# Phase pair counts
ppc = [ph.ms_pair_count for ph in meta.phases]
print(f'Phase pair counts: {ppc}')
print(f'Sum: {sum(ppc)}')

# Create compiler and decompose (no routing)
cfg = QCCDWiseArch(m=2, n=3, k=2)
arch = WISEArchitecture(wise_config=cfg, add_spectators=True, compact_clustering=True)
comp = TrappedIonCompiler(arch, is_wise=True, wise_config=cfg)
native = comp.decompose_to_native(ideal, qec_metadata=meta)

# Check parallelPairs
if hasattr(comp, 'parallelPairs'):
    pp = comp.parallelPairs
    print(f'\nparallelPairs: {len(pp)} rounds')
    total_pairs = sum(len(r) for r in pp)
    print(f'total pairs: {total_pairs}')
    print(f'sum(phase_pair_counts)={sum(ppc)} vs len(parallelPairs)={len(pp)}')
    
    # Group by phase
    idx = 0
    for pi, ph in enumerate(meta.phases):
        cnt = ph.ms_pair_count
        if cnt > 0:
            rounds_in_phase = pp[idx:idx+cnt]
            pairs_in_phase = sum(len(r) for r in rounds_in_phase)
            print(f'  phase[{pi}] {ph.phase_type:25s}: {cnt} rounds, {pairs_in_phase} pairs')
            for ri, rnd in enumerate(rounds_in_phase[:12]):
                # Check if any pair involves bridge qubits
                bridge_involved = False
                bridge_qubits = {21, 22, 23}
                for pair in rnd:
                    if any(q in bridge_qubits for q in pair):
                        bridge_involved = True
                        break
                tag = " [BRIDGE]" if bridge_involved else " [EC]"
                print(f'    round[{ri:2d}]: {len(rnd)} pairs{tag}  {sorted(rnd)[:4]}')
            idx += cnt
        else:
            print(f'  phase[{pi}] {ph.phase_type:25s}: 0 rounds')

# Bridge info
print(f'\nBridge ancillas:')
bridge_qubits = set()
if hasattr(alloc, 'bridge_ancillas') and alloc.bridge_ancillas:
    for bi in alloc.bridge_ancillas:
        bridge_qubits.add(bi[0])
        print(f'  qubit={bi[0]}, pos={bi[1]}, label={bi[2]}')

# Bridge CX in stim circuit
print(f'\nBridge CX in stim (by tick):')
tick = 0
for inst in ideal.flattened():
    if inst.name == 'TICK': tick += 1
    elif inst.name in ('CX','CNOT'):
        tgts = inst.targets_copy()
        for j in range(0, len(tgts), 2):
            q1, q2 = tgts[j].value, tgts[j+1].value
            if q1 in bridge_qubits or q2 in bridge_qubits:
                print(f'  tick {tick:3d}: CX({q1},{q2})')

# Check toMoveOps for bridge ion assignments
if hasattr(comp, 'toMoveOps'):
    print(f'\ntoMoveOps: {len(comp.toMoveOps)} rounds')
    bridge_ions = set()
    # q2i mapping
    q2i = {}
    for ba in meta.block_allocations:
        for q in list(ba.data_qubits) + list(ba.x_ancilla_qubits) + list(ba.z_ancilla_qubits):
            q2i[q] = q + 1
    for bq in bridge_qubits:
        q2i[bq] = bq + 1
        bridge_ions.add(bq + 1)
    
    for ri, rnd in enumerate(comp.toMoveOps):
        ms_ops = [op for op in rnd if type(op).__name__ == 'TwoQubitMSGate']
        has_bridge = False
        for op in ms_ops:
            ions = set(qp.ion.idx for qp in op.qubit_parts)
            if ions & bridge_ions:
                has_bridge = True
                break
        tag = " [BRIDGE]" if has_bridge else " [EC]"
        if ri < 40:
            print(f'  round[{ri:2d}]: {len(rnd)} ops, {len(ms_ops)} MS{tag}')
            if has_bridge:
                for op in ms_ops[:3]:
                    ions = tuple(qp.ion.idx for qp in op.qubit_parts)
                    print(f'           MS ions={ions}')

# Check qubit-to-ion mapping 
print(f'\nQubit-to-ion mapping for bridge qubits:')
for bq in sorted(bridge_qubits):
    # Check if the compiler actually assigned this ion
    found = False
    for ion in arch._ions:
        if hasattr(ion, 'idx') and ion.idx == bq + 1:
            trap = ion.currentTrap
            pos = f'row={trap.row},col={trap.col}' if trap else 'NO TRAP'
            print(f'  qubit {bq} -> ion {ion.idx}: {pos}')
            found = True
            break
    if not found:
        print(f'  qubit {bq} -> ion {bq+1}: NOT FOUND in arch._ions')

print('\nDONE')
