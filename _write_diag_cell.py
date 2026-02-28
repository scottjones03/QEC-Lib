import json

with open('notebooks/trapped_ion_demo.ipynb') as f:
    nb = json.load(f)

diag_code = '''"""DIAGNOSTIC: Trace ion 28 through animation replay."""
from qectostim.experiments.hardware_simulation.trapped_ion.utils.qccd_operations import GlobalReconfigurations
from qectostim.experiments.hardware_simulation.trapped_ion.utils.qccd_operations_on_qubits import QubitOperation

TARGET_ION = 28
WINDOW_START = 118
WINDOW_END = 132

arch_g.resetArrangement()
arch_g.refreshGraph()

print(f"=== Diagnostic: tracing ion {TARGET_ION} through {len(batches_g)} steps ===\\n")

ion_obj = arch_g.ions[TARGET_ION]
print(f"Initial: ion {TARGET_ION} in {type(ion_obj.parent).__name__} idx={ion_obj.parent.idx}\\n")

for i, op in enumerate(batches_g):
    op_type = type(op).__name__
    is_reconfig = isinstance(op, GlobalReconfigurations)
    is_qubit_op = isinstance(op, QubitOperation)
    pre_parent = ion_obj.parent
    pre_idx = pre_parent.idx if pre_parent else None

    expected_trap_idx = None
    involves_target = False
    if is_qubit_op:
        et = getattr(op, "_trap", None)
        expected_trap_idx = et.idx if et else None
        involves_target = any(ion.idx == TARGET_ION for ion in op.ions)

    error = None
    try:
        op.run()
        if hasattr(arch_g, "refreshGraph"):
            arch_g.refreshGraph()
    except Exception as exc:
        error = str(exc)
        if hasattr(arch_g, "refreshGraph"):
            try:
                arch_g.refreshGraph()
            except Exception:
                pass

    post_parent = ion_obj.parent
    post_idx = post_parent.idx if post_parent else None
    moved = (pre_idx != post_idx)
    in_window = WINDOW_START <= i <= WINDOW_END

    if error and str(TARGET_ION) in error:
        print(f"*** STEP {i} FAILED: {op_type}: {error}")
        print(f"    ion {TARGET_ION} pre: trap {pre_idx}")
        print(f"    ion {TARGET_ION} post: trap {post_idx}")
        if is_qubit_op:
            print(f"    expected trap: {expected_trap_idx}")
            print(f"    op ions: {[ion.idx for ion in op.ions]}")
        print()
    elif in_window:
        label = getattr(op, "label", "") or ""
        info = f"  Step {i}: {op_type}"
        if label:
            info += f" [{label}]"
        if is_reconfig:
            info += f" | ion {TARGET_ION}: trap {pre_idx} -> {post_idx}"
        elif involves_target:
            info += f" | ion {TARGET_ION} in trap {pre_idx}, expected={expected_trap_idx}"
            if error:
                info += f" *** FAILED: {error}"
        else:
            info += f" | ion {TARGET_ION} in trap {post_idx}"
        print(info)
    elif moved and is_reconfig:
        print(f"  Step {i}: GlobalReconfig moved ion {TARGET_ION}: trap {pre_idx} -> {post_idx}")
    elif error:
        print(f"  Step {i}: FAILED: {error[:80]}")

print(f"\\n=== Done: {len(batches_g)} steps replayed ===")'''

# Convert to list of lines with newlines
lines = diag_code.split('\n')
source = [line + '\n' for line in lines[:-1]]
source.append(lines[-1])  # last line without trailing newline

nb['cells'][22]['source'] = source
nb['cells'][22]['outputs'] = []

with open('notebooks/trapped_ion_demo.ipynb', 'w') as f:
    json.dump(nb, f, indent=1)

print(f"Done: cell 22 now has {len(source)} lines of diagnostic code")
