"""Patch notebook cell 22 (index 22) with improved diagnostic v2."""
import json, sys

NB = "notebooks/trapped_ion_demo.ipynb"
CELL_IDX = 22  # cell #VSC-0d4ce771 (the empty/diagnostic cell)

NEW_SOURCE = r'''"""DIAGNOSTIC v2: Trace ion 28 — show EVERY move and all errors."""
from qectostim.experiments.hardware_simulation.trapped_ion.utils.qccd_operations import (
    GlobalReconfigurations, ParallelOperation,
)
from qectostim.experiments.hardware_simulation.trapped_ion.utils.qccd_operations_on_qubits import QubitOperation

TARGET_ION = 28
WINDOW = range(118, 133)

arch_g.resetArrangement()
arch_g.refreshGraph()

ion_obj = arch_g.ions[TARGET_ION]
print(f"=== Tracing ion {TARGET_ION} through {len(batches_g)} steps ===")
print(f"Initial: trap {ion_obj.parent.idx}\n")

def _inner_types(op):
    kids = getattr(op, '_operations', getattr(op, 'operations', []))
    if kids:
        return [type(k).__name__ for k in kids]
    return [type(op).__name__]

def _has_reconfig(op):
    kids = getattr(op, '_operations', getattr(op, 'operations', [op]))
    return any(isinstance(k, GlobalReconfigurations) for k in kids)

def _ion28_inner_ops(op):
    kids = getattr(op, '_operations', getattr(op, 'operations', [op]))
    out = []
    for k in kids:
        if isinstance(k, QubitOperation):
            if any(ion.idx == TARGET_ION for ion in k.ions):
                out.append(k)
    return out

errors_found = []
for i, op in enumerate(batches_g):
    pre_trap = ion_obj.parent.idx if ion_obj.parent else None

    error = None
    try:
        op.run()
        if hasattr(arch_g, 'refreshGraph'):
            arch_g.refreshGraph()
    except Exception as exc:
        error = str(exc)
        try:
            arch_g.refreshGraph()
        except Exception:
            pass

    post_trap = ion_obj.parent.idx if ion_obj.parent else None
    moved = pre_trap != post_trap
    in_window = i in WINDOW

    ion28_ops = _ion28_inner_ops(op)

    if error and str(TARGET_ION) in error:
        errors_found.append(i)
        exp_traps = [getattr(iop, '_trap', None) for iop in ion28_ops]
        exp_str = [t.idx if t else '?' for t in exp_traps]
        print(f"*** STEP {i} FAILED: {error}")
        print(f"    ion {TARGET_ION}: trap {pre_trap} (expected {exp_str})")
        print(f"    inner_types: {_inner_types(op)}")
        for iop in ion28_ops:
            print(f"    op_ions: {[ion.idx for ion in iop.ions]}, trap={getattr(iop,'_trap',None) and getattr(iop,'_trap').idx}")
        print()
    elif moved:
        rc = " [RECONFIG]" if _has_reconfig(op) else ""
        print(f"  Step {i}: MOVED ion {TARGET_ION}: trap {pre_trap} -> {post_trap}{rc}  ({_inner_types(op)})")
    elif in_window:
        lbl = getattr(op, 'label', '') or ''
        extra = ""
        if ion28_ops:
            extra = f" [ion28 expected_trap={[getattr(o,'_trap',None) and getattr(o,'_trap').idx for o in ion28_ops]}]"
        print(f"  Step {i}: {lbl:40s} trap={post_trap}{extra}  ({_inner_types(op)})")

print(f"\n=== Done: {len(batches_g)} steps, {len(errors_found)} errors at {errors_found} ===")
'''

with open(NB) as f:
    nb = json.load(f)

cell = nb['cells'][CELL_IDX]
print(f"Patching cell {CELL_IDX}: id={cell.get('id','?')}, current_len={len(''.join(cell['source']))}")

# Convert source to list of lines with newlines
lines = NEW_SOURCE.split('\n')
cell['source'] = [line + '\n' for line in lines[:-1]] + [lines[-1]]

with open(NB, 'w') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("Done — cell patched successfully")
