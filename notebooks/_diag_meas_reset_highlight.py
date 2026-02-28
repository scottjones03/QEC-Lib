"""Diagnostic: trace measurement/reset stim highlighting in detail."""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), '..', 'src')))
os.environ["WISE_INPROCESS_LIMIT"] = "999999999"

from qectostim.codes.surface.rotated_surface import RotatedSurfaceCode
from qectostim.experiments.memory import CSSMemoryExperiment
from qectostim.experiments.hardware_simulation.trapped_ion.utils import (
    WISEArchitecture, TrappedIonCompiler,
)
from qectostim.experiments.hardware_simulation.trapped_ion.utils.qccd_nodes import QCCDWiseArch
from qectostim.experiments.hardware_simulation.trapped_ion.compiler.routing_config import WISERoutingConfig
from qectostim.experiments.hardware_simulation.trapped_ion.viz.stim_panel import parse_stim_for_sidebar
from qectostim.experiments.hardware_simulation.trapped_ion.viz.display import (
    _gate_kind, _extract_active_ions,
)

# ── Build and compile ──
dis = 2
code = RotatedSurfaceCode(distance=dis)
mem = CSSMemoryExperiment(code=code, rounds=dis, noise_model=None, basis='z')
ideal = mem.to_stim()

print("=== IDEAL STIM CIRCUIT ===")
print(ideal)
print()

wise_cfg = QCCDWiseArch(m=2, n=3, k=2)
arch = WISEArchitecture(wise_config=wise_cfg, add_spectators=True, compact_clustering=True)
compiler = TrappedIonCompiler(arch, is_wise=True, wise_config=wise_cfg)
compiler.routing_kwargs = dict(
    routing_config=WISERoutingConfig.default(lookahead=1, subgridsize=(12, 12, 1)),
)
compiled = compiler.compile(ideal)
batches = compiled.scheduled.batches
print(f"Compiled: {len(batches)} steps\n")

# ── Sidebar entries ──
sidebar = parse_stim_for_sidebar(ideal)
print("=== SIDEBAR ENTRIES ===")
for i, e in enumerate(sidebar):
    print(f"  [{i:2d}] kind={e.kind:10s} gate={e.gate or '':6s} qubits={e.qubits}  text={e.text!r}")
print()

# Gate entry indices
gate_entry_indices = {i for i, e in enumerate(sidebar) if e.kind == "gate"}
print(f"Gate entry indices: {sorted(gate_entry_indices)}\n")

# ── Per-operation _stim_origin ──
print("=== PER-STEP NATIVE OPS AND _stim_origin ===")

def collect_sub_ops(op):
    cls = type(op).__name__
    if cls == "ParallelOperation":
        result = []
        for sub in getattr(op, "operations", []):
            result.extend(collect_sub_ops(sub))
        return result
    return [op]

for si, batch in enumerate(batches):
    gk = _gate_kind(batch)
    sub_ops = collect_sub_ops(batch)
    origins = []
    for sub in sub_ops:
        origin = getattr(sub, "_stim_origin", None)
        cls = type(sub).__name__
        origins.append((cls, origin))

    if gk is None:
        label = "transport"
    else:
        label = gk

    # Only show non-transport steps
    if gk is not None:
        origin_set = {o for _, o in origins if o is not None and o >= 0}
        origin_entries = {}
        for o in origin_set:
            if o < len(sidebar):
                origin_entries[o] = (sidebar[o].gate, sidebar[o].qubits)
        print(f"  Step {si+1:3d}: gk={label:10s} sub_ops={len(sub_ops)}")
        for cls, o in origins:
            if o is not None and o >= 0 and o < len(sidebar):
                e = sidebar[o]
                print(f"           {cls:20s} _stim_origin={o:3d} → gate={e.gate!r:6s} qubits={e.qubits}")
            else:
                print(f"           {cls:20s} _stim_origin={o}")
        print()

# ── Now run the actual stim mapping ──
print("=== STIM MAPPING (via _build_stim_mapping) ===")

from qectostim.experiments.hardware_simulation.trapped_ion.viz.animation import _build_stim_mapping

# Build gate_kinds and active_ions
gate_kinds = [None]  # step 0 = initial
active_ions_list = [set()]
for batch in batches:
    gate_kinds.append(_gate_kind(batch))
    active_ions_list.append(_extract_active_ions(batch))

from qectostim.experiments.hardware_simulation.trapped_ion.demo.run import build_viz_mappings
ion_roles, p2l, remap = build_viz_mappings(compiler)

sidebar2, tick_svg_cache, stim_lines_per_step, stim_tick_per_step, n_ticks = _build_stim_mapping(
    ideal,
    len(batches),
    gate_kinds,
    active_ions_list,
    remap,
    p2l,
    operations=batches,
)

print(f"n_ticks = {n_ticks}\n")

# Focus on measurement and reset steps
print("=== ALL STEPS WITH STIM MAPPING (focus on M/R) ===")
for si in range(len(batches) + 1):
    gk = gate_kinds[si] if si < len(gate_kinds) else None
    tick = stim_tick_per_step[si]
    entries = stim_lines_per_step[si]
    
    gk_str = gk or "transport/init"
    entry_details = ""
    if entries:
        for eidx in sorted(entries):
            if eidx < len(sidebar2):
                e = sidebar2[eidx]
                entry_details += f" [{eidx}:{e.gate}({e.qubits})]"
    
    is_meas_reset = gk in ("measure", "reset")
    marker = " <<<" if is_meas_reset else ""
    print(f"  Step {si:3d}: gk={gk_str:12s} tick={tick}  entries={entries}{entry_details}{marker}")

print("\n=== SUMMARY ===")
meas_steps = [si for si in range(len(batches)+1) if (gate_kinds[si] if si < len(gate_kinds) else None) == "measure"]
reset_steps = [si for si in range(len(batches)+1) if (gate_kinds[si] if si < len(gate_kinds) else None) == "reset"]
print(f"Measurement steps: {meas_steps}")
print(f"Reset steps: {reset_steps}")

# Check if measurement entries point to M gates in sidebar
print("\nMeasurement step entry validation:")
for si in meas_steps:
    entries = stim_lines_per_step[si]
    if entries:
        for eidx in sorted(entries):
            if eidx < len(sidebar2):
                e = sidebar2[eidx]
                is_meas_entry = e.gate and e.gate.upper() in {"M", "MR", "MX", "MY", "MZ", "MRX", "MRY", "MRZ"}
                status = "OK" if is_meas_entry else f"WRONG (gate={e.gate!r})"
                print(f"  Step {si}: entry {eidx} → gate={e.gate!r} qubits={e.qubits} → {status}")
    else:
        print(f"  Step {si}: NO ENTRIES!")

print("\nReset step entry validation:")
for si in reset_steps:
    entries = stim_lines_per_step[si]
    if entries:
        for eidx in sorted(entries):
            if eidx < len(sidebar2):
                e = sidebar2[eidx]
                is_reset_entry = e.gate and e.gate.upper() in {"R", "RX", "RY", "RZ"}
                status = "OK" if is_reset_entry else f"WRONG (gate={e.gate!r})"
                print(f"  Step {si}: entry {eidx} → gate={e.gate!r} qubits={e.qubits} → {status}")
    else:
        print(f"  Step {si}: NO ENTRIES!")
