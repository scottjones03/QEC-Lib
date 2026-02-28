"""Patch trapped_ion_demo.ipynb: add Phase-Aware Gadget Routing + d=7 cells."""
import json
import uuid
import sys

NB_PATH = "notebooks/trapped_ion_demo.ipynb"

with open(NB_PATH) as f:
    nb = json.load(f)

print(f"Starting with {len(nb['cells'])} cells")


def mk_code(source_str: str) -> dict:
    lines = source_str.split("\n")
    src = [l + "\n" for l in lines[:-1]] + [lines[-1]] if lines else []
    return {
        "cell_type": "code",
        "execution_count": None,
        "id": uuid.uuid4().hex[:8],
        "metadata": {},
        "outputs": [],
        "source": src,
    }


def mk_md(source_str: str) -> dict:
    lines = source_str.split("\n")
    src = [l + "\n" for l in lines[:-1]] + [lines[-1]] if lines else []
    return {
        "cell_type": "markdown",
        "id": uuid.uuid4().hex[:8],
        "metadata": {},
        "source": src,
    }


# ── Cells to insert (after Pipeline Ref header, before Pipeline Ref) ──
# We insert just before the Pipeline Reference markdown (last cell).

# 1. Phase-Aware markdown header
md_phase_aware = mk_md(
    "---\n"
    "## Phase-Aware Gadget Routing (Two-Level Spatial Slicing)\n"
    "\n"
    "End-to-end demonstration of the **gadget routing pipeline** from\n"
    "`GADGET_COMPILATION_SPEC.md`.\n"
    "\n"
    "The pipeline decomposes a fault-tolerant gadget experiment into temporal\n"
    "**phases** (EC rounds + gadget round), then routes each phase on the\n"
    "WISE grid using two levels of spatial slicing:\n"
    "\n"
    "| Level | Scope | Function |\n"
    "|-------|-------|----------|\n"
    "| **Level 1** | Block-level partitioning | `partition_grid_for_blocks()` — assigns disjoint sub-grids per code block |\n"
    "| **Level 2** | Patch-and-route | `_patch_and_route()` — SAT-based WISE routing within each sub-grid |\n"
    "\n"
    "**Key building blocks** exercised below:\n"
    "- `FaultTolerantGadgetExperiment` → ideal circuit + `QECMetadata`\n"
    "- `decompose_into_phases()` → `PhaseRoutingPlan` list\n"
    "- `route_full_experiment()` → per-phase routing with EC round caching\n"
    "- `compute_schedule_timing()` → analytical execution & reconfiguration time"
)

# 2. Step 1 – Build experiment
code_step1 = mk_code(
    '"""Step 1: Build FaultTolerantGadgetExperiment and extract QECMetadata."""\n'
    "import time as _time\n"
    "from qectostim.experiments.ft_gadget_experiment import FaultTolerantGadgetExperiment\n"
    "from qectostim.gadgets.transversal_cnot import TransversalCNOTGadget\n"
    "from qectostim.experiments.hardware_simulation.trapped_ion.utils.gadget_routing import (\n"
    "    partition_grid_for_blocks,\n"
    "    decompose_into_phases,\n"
    "    route_full_experiment,\n"
    "    FullExperimentResult,\n"
    ")\n"
    "\n"
    "# Parameters\n"
    "d_gadget = 2   # code distance (keep small for demo speed)\n"
    "k = 2          # trap capacity (ions per trap)\n"
    "n_rounds = d_gadget\n"
    "\n"
    "# Build the fault-tolerant experiment\n"
    "gadget_cnot = TransversalCNOTGadget()\n"
    "code_d2 = RotatedSurfaceCode(distance=d_gadget)\n"
    "ft_exp = FaultTolerantGadgetExperiment(\n"
    "    codes=[code_d2],\n"
    "    gadget=gadget_cnot,\n"
    "    noise_model=None,\n"
    "    num_rounds_before=n_rounds,\n"
    "    num_rounds_after=n_rounds,\n"
    ")\n"
    "\n"
    "# Generate ideal stim circuit -- this populates qec_metadata\n"
    "ideal_gadget = ft_exp.to_stim()\n"
    "qec_meta = ft_exp.qec_metadata\n"
    "qubit_alloc = ft_exp._unified_allocation\n"
    "\n"
    'print(f"Ideal circuit:  {len(ideal_gadget)} instructions, "\n'
    '      f"{ideal_gadget.num_qubits} qubits")\n'
    'print(f"Code blocks:    {[ba.block_name for ba in qec_meta.block_allocations]}")\n'
    'print(f"Phases:         {len(qec_meta.phases)}")\n'
    "for i, ph in enumerate(qec_meta.phases):\n"
    '    print(f"  [{i}] {ph.phase_type!s:10s}  blocks={ph.active_blocks}")'
)

# 3. Step 2 – Level 1 partition + decompose
code_step2 = mk_code(
    '"""Step 2: Level 1 -- Partition grid into per-block sub-grids."""\n'
    "sub_grids = partition_grid_for_blocks(qec_meta, qubit_alloc, k)\n"
    "\n"
    'print("Block sub-grid allocations (Level 1 spatial slicing):")\n'
    "for name, sg in sub_grids.items():\n"
    "    layout_shape = sg.initial_layout.shape if sg.initial_layout is not None else 'None'\n"
    '    print(f"  {name:12s}  rows={sg.n_rows}, traps/row={sg.n_cols}, "\n'
    '          f"ions/row={sg.n_cols * k}, layout={layout_shape}")\n'
    "print()\n"
    "\n"
    "# Decompose into phase routing plans\n"
    "qubit_to_ion = {q: q + 1 for ba in qec_meta.block_allocations\n"
    "                for q in list(ba.data_qubits) + list(ba.x_ancilla_qubits) + list(ba.z_ancilla_qubits)}\n"
    "\n"
    "plans = decompose_into_phases(\n"
    "    qec_meta, gadget_cnot, qubit_alloc, sub_grids, qubit_to_ion, k\n"
    ")\n"
    "\n"
    'print(f"Routing plans: {len(plans)}")\n'
    "for i, p in enumerate(plans):\n"
    "    rounds_info = len(p.ms_pairs_per_round) if p.ms_pairs_per_round else 0\n"
    '    print(f"  [{i}] type={p.phase_type:8s}  blocks={p.interacting_blocks}  "\n'
    '          f"ms_rounds={rounds_info}  cacheable={p.round_signature is not None}")'
)

# 4. Step 3 – Run full routing (d=2)
code_step3 = mk_code(
    '"""Step 3: Run full phase-aware routing (Level 2 patch-and-route per phase)."""\n'
    "import logging\n"
    "logging.basicConfig(level=logging.INFO)\n"
    "\n"
    "t0 = _time.perf_counter()\n"
    "full_result = route_full_experiment(\n"
    "    qec_meta=qec_meta,\n"
    "    gadget=gadget_cnot,\n"
    "    qubit_allocation=qubit_alloc,\n"
    "    k=k,\n"
    "    subgridsize=(4, 3, 0),   # (width, height, increment)\n"
    "    base_pmax_in=1,\n"
    "    lookahead=2,\n"
    "    max_inner_workers=1,     # serial for notebook stability\n"
    ")\n"
    "wall = _time.perf_counter() - t0\n"
    "\n"
    'print(f"\\n=== Phase-Aware Routing Results ===")\n'
    'print(f"Wall time:           {wall:.1f} s")\n'
    'print(f"Execution time:      {full_result.total_exec_time:.6f} s")\n'
    'print(f"Reconfiguration:     {full_result.total_reconfig_time:.6f} s")\n'
    'print(f"Phases routed:       {full_result.total_phases}")\n'
    'print(f"  Cached (replayed): {full_result.cached_phases}")\n'
    'print(f"  Fresh (SAT):       {full_result.total_phases - full_result.cached_phases}")\n'
    'print(f"MS rounds routed:    {full_result.ms_rounds_routed}")\n'
    'print(f"MS rounds replayed:  {full_result.ms_rounds_replayed}")\n'
    'print(f"Schedule steps:      {len(full_result.total_schedule)}")\n'
    "\n"
    'print(f"\\n--- Per-Phase Breakdown ---")\n'
    "for i, pr in enumerate(full_result.phase_results):\n"
    '    cache_tag = " (cached)" if pr.from_cache else ""\n'
    '    print(f"  Phase {i}: type={pr.phase_type:8s}  "\n'
    '          f"exec={pr.exec_time:.6f}s  reconfig={pr.reconfig_time:.6f}s"\n'
    '          f"  steps={len(pr.schedule)}{cache_tag}")'
)

# 5. d=3 run_single_gadget_config
code_d3_run = mk_code(
    "import time, logging, numpy as np\n"
    "logging.basicConfig(level=logging.INFO)\n"
    "\n"
    "from qectostim.experiments.hardware_simulation.trapped_ion.utils.best_effort_compilation_WISE import (\n"
    "    run_single_gadget_config,\n"
    ")\n"
    "from qectostim.codes.surface.rotated_surface import RotatedSurfaceCode\n"
    "from qectostim.gadgets.transversal_cnot import TransversalCNOTGadget\n"
    "from qectostim.gadgets.knill_ec import KnillECGadget\n"
    "from qectostim.gadgets.css_surgery_cnot import CSSSurgeryCNOTGadget\n"
    "\n"
    '# ---------- d=3  TransversalCNOT  (2 blocks, 1 gadget phase) ----------\n'
    "code_3 = RotatedSurfaceCode(distance=3)\n"
    'gadget_tcnot = TransversalCNOTGadget(control_state="0", target_state="0")\n'
    "\n"
    'print("\\u25b6 d=3 TransversalCNOT \\u2026")\n'
    "t0 = time.perf_counter()\n"
    "exec_t, comp_t, res, reconf_t = run_single_gadget_config(\n"
    "    gadget=gadget_tcnot, code=code_3,\n"
    "    lookahead=2, subgrid_width=4, subgrid_height=3,\n"
    "    subgrid_increment=0, trap_capacity=2, base_pmax_in=1,\n"
    "    gate_improvements=[1.0], num_shots=100, rounds=2,\n"
    "    show_progress=False, max_inner_workers=1,\n"
    ")\n"
    "wall = time.perf_counter() - t0\n"
    'print(f"  Wall time:          {wall:.1f}s")\n'
    'print(f"  Phase-aware:        {res.get(\'phase_aware\')}")\n'
    'print(f"  Blocks:             {res.get(\'num_blocks\')}  {res.get(\'block_names\')}")\n'
    'print(f"  Phases routed:      {res.get(\'total_phases\')}")\n'
    'print(f"  Cached phases:      {res.get(\'cached_phases\')}")\n'
    'print(f"  MS rounds routed:   {res.get(\'ms_rounds_routed\')}")\n'
    'print(f"  MS rounds replayed: {res.get(\'ms_rounds_replayed\')}")\n'
    'print(f"  Exec time (\\u03bcs):     {exec_t}")\n'
    'print(f"  Reconfig time (\\u03bcs): {reconf_t}")\n'
    'print(f"  Logical error rate: {res.get(\'LogicalErrorRates\')}")\n'
    'print("  \\u2713 PASS" if not np.isnan(exec_t) or res.get("phase_aware") else "  \\u2717 FAIL")'
)

# 6. d=7 markdown header
md_d7_header = mk_md(
    "---\n"
    "### d = 7 Two-Level Spatial Slicing — Analytical Breakdown\n"
    "\n"
    "At **d = 7** each rotated-surface-code block needs **97 data + ancilla qubits**.\n"
    "Level 1 partitions the grid into **disjoint per-block sub-grids**, and\n"
    "Level 2 tiles each sub-grid into small **SAT-solvable patches**.\n"
    "\n"
    "Below we build the experiment and inspect the decomposition *without*\n"
    "running the slow SAT solver, then compare the slicing geometry across\n"
    "d ∈ {2, 3, 5, 7}."
)

# 7. d=7 Level 1 partition + phase decomposition
code_d7_partition = mk_code(
    '"""d=7: Level 1 partition + Phase decomposition (no SAT solving)."""\n'
    "import math\n"
    "from qectostim.experiments.ft_gadget_experiment import FaultTolerantGadgetExperiment\n"
    "from qectostim.gadgets.transversal_cnot import TransversalCNOTGadget\n"
    "from qectostim.codes.surface.rotated_surface import RotatedSurfaceCode\n"
    "from qectostim.experiments.hardware_simulation.trapped_ion.utils.gadget_routing import (\n"
    "    allocate_block_regions,\n"
    "    partition_grid_for_blocks,\n"
    "    decompose_into_phases,\n"
    ")\n"
    "\n"
    "d7 = 7\n"
    "k7 = 2\n"
    "code_d7 = RotatedSurfaceCode(distance=d7)\n"
    "gadget_d7 = TransversalCNOTGadget()\n"
    "\n"
    "ft_d7 = FaultTolerantGadgetExperiment(\n"
    "    codes=[code_d7],\n"
    "    gadget=gadget_d7,\n"
    "    noise_model=None,\n"
    "    num_rounds_before=d7,\n"
    "    num_rounds_after=d7,\n"
    ")\n"
    "ideal_d7 = ft_d7.to_stim()\n"
    "meta_d7 = ft_d7.qec_metadata\n"
    "alloc_d7 = ft_d7._unified_allocation\n"
    "\n"
    'print(f"=== d = {d7}  TransversalCNOT ===")\n'
    'print(f"Ideal circuit: {len(ideal_d7)} instructions, {ideal_d7.num_qubits} qubits")\n'
    'print(f"Code blocks:   {[ba.block_name for ba in meta_d7.block_allocations]}")\n'
    'print(f"Phases:        {len(meta_d7.phases)}")\n'
    "for i, ph in enumerate(meta_d7.phases):\n"
    '    print(f"  [{i}] {ph.phase_type!s:20s}  blocks={ph.active_blocks}")\n'
    "\n"
    "# Level 1: per-block sub-grids\n"
    "sub_grids_d7 = partition_grid_for_blocks(meta_d7, alloc_d7, k7)\n"
    'print(f"\\n--- Level 1: Block Sub-Grid Allocations ---")\n'
    "for name, sg in sub_grids_d7.items():\n"
    "    traps = sg.n_rows * sg.n_cols\n"
    "    ions = traps * k7\n"
    '    print(f"  {name:12s}  region={sg.grid_region}  "\n'
    '          f"shape={sg.n_rows}\\u00d7{sg.n_cols} traps  "\n'
    '          f"= {traps} traps ({ions} ion slots) for {len(sg.ion_indices)} ions")\n'
    "\n"
    "# Phase decomposition\n"
    "q2i_d7 = {q: q + 1 for ba in meta_d7.block_allocations\n"
    "           for q in list(ba.data_qubits) + list(ba.x_ancilla_qubits) + list(ba.z_ancilla_qubits)}\n"
    "plans_d7 = decompose_into_phases(\n"
    "    meta_d7, gadget_d7, alloc_d7, sub_grids_d7, q2i_d7, k7\n"
    ")\n"
    "\n"
    'print(f"\\n--- Phase Routing Plans ({len(plans_d7)} phases) ---")\n'
    "cached_count = sum(1 for p in plans_d7 if p.identical_to_phase is not None)\n"
    "fresh_count = len(plans_d7) - cached_count\n"
    'print(f"  Fresh (SAT-routed): {fresh_count}")\n'
    'print(f"  Cached (replayed):  {cached_count}")\n'
    "for i, p in enumerate(plans_d7):\n"
    "    nrounds = len(p.ms_pairs_per_round) if p.ms_pairs_per_round else 0\n"
    '    cache_info = f"  (copy of phase {p.identical_to_phase})" if p.identical_to_phase is not None else ""\n'
    '    print(f"  [{i:2d}] {p.phase_type:20s}  blocks={p.interacting_blocks}  "\n'
    '          f"ms_rounds={nrounds}{cache_info}")'
)

# 8. d=7 Level 2 patch tiling
code_d7_patches = mk_code(
    '"""d=7: Level 2 patch tiling analysis."""\n'
    "from qectostim.experiments.hardware_simulation.trapped_ion.compiler.qccd_WISE_ion_route import (\n"
    "    _generate_patch_regions,\n"
    ")\n"
    "\n"
    "subgrid_w, subgrid_h = 4, 3  # default patch dimensions\n"
    'print(f"Patch dimensions: {subgrid_w} cols \\u00d7 {subgrid_h} rows\\n")\n'
    "\n"
    "for name, sg in sub_grids_d7.items():\n"
    "    n_r, n_c = sg.n_rows, sg.n_cols\n"
    "    print(f\"Block '{name}': sub-grid = {n_r} rows \\u00d7 {n_c} cols\")\n"
    "\n"
    "    # Tiling 0: no offset (standard)\n"
    "    regions_0 = _generate_patch_regions(n_r, n_c, subgrid_h, subgrid_w, 0, 0)\n"
    "    # Tiling 1: vertical offset\n"
    "    regions_v = _generate_patch_regions(n_r, n_c, subgrid_h, subgrid_w, subgrid_h // 2, 0)\n"
    "    # Tiling 2: horizontal offset\n"
    "    regions_h = _generate_patch_regions(n_r, n_c, subgrid_h, subgrid_w, 0, subgrid_w // 2)\n"
    "    # Tiling 3: both offsets\n"
    "    regions_vh = _generate_patch_regions(n_r, n_c, subgrid_h, subgrid_w, subgrid_h // 2, subgrid_w // 2)\n"
    "\n"
    '    print(f"  Tiling 0 (no offset):       {len(regions_0):2d} patches  {regions_0[:3]}...")\n'
    '    print(f"  Tiling 1 (vert offset):     {len(regions_v):2d} patches  {regions_v[:3]}...")\n'
    '    print(f"  Tiling 2 (horiz offset):    {len(regions_h):2d} patches  {regions_h[:3]}...")\n'
    '    print(f"  Tiling 3 (both offsets):    {len(regions_vh):2d} patches  {regions_vh[:3]}...")\n'
    '    print(f"  \\u2192 Each SAT call solves a \\u2264{subgrid_h}\\u00d7{subgrid_w} sub-problem")\n'
    '    print(f"    instead of the full {n_r}\\u00d7{n_c} grid")\n'
    "    print()\n"
    "\n"
    "total_ions_d7 = sum(len(sg.ion_indices) for sg in sub_grids_d7.values())\n"
    'print(f"Total ions across all blocks: {total_ions_d7}")\n'
    'print(f"For comparison, d=2 used ~{2 * (2*2 + (2-1)**2)} ions total")'
)

# 9. Scaling comparison
code_scaling = mk_code(
    '"""Scaling comparison across d = 2, 3, 5, 7."""\n'
    "import math\n"
    "from qectostim.experiments.ft_gadget_experiment import FaultTolerantGadgetExperiment\n"
    "from qectostim.gadgets.transversal_cnot import TransversalCNOTGadget\n"
    "from qectostim.codes.surface.rotated_surface import RotatedSurfaceCode\n"
    "from qectostim.experiments.hardware_simulation.trapped_ion.utils.gadget_routing import (\n"
    "    partition_grid_for_blocks,\n"
    "    decompose_into_phases,\n"
    ")\n"
    "from qectostim.experiments.hardware_simulation.trapped_ion.compiler.qccd_WISE_ion_route import (\n"
    "    _generate_patch_regions,\n"
    ")\n"
    "\n"
    "k_sc = 2\n"
    "pw, ph_patch = 4, 3  # patch dims\n"
    "\n"
    "rows = []\n"
    "for d_sc in [2, 3, 5, 7]:\n"
    "    code_sc = RotatedSurfaceCode(distance=d_sc)\n"
    "    gadget_sc = TransversalCNOTGadget()\n"
    "    ft_sc = FaultTolerantGadgetExperiment(\n"
    "        codes=[code_sc], gadget=gadget_sc, noise_model=None,\n"
    "        num_rounds_before=d_sc, num_rounds_after=d_sc,\n"
    "    )\n"
    "    ideal_sc = ft_sc.to_stim()\n"
    "    meta_sc = ft_sc.qec_metadata\n"
    "    alloc_sc = ft_sc._unified_allocation\n"
    "\n"
    "    sg_map = partition_grid_for_blocks(meta_sc, alloc_sc, k_sc)\n"
    "    q2i_sc = {q: q + 1 for ba in meta_sc.block_allocations\n"
    "              for q in list(ba.data_qubits) + list(ba.x_ancilla_qubits) + list(ba.z_ancilla_qubits)}\n"
    "    plans_sc = decompose_into_phases(meta_sc, gadget_sc, alloc_sc, sg_map, q2i_sc, k_sc)\n"
    "\n"
    "    # Get first block's grid for patch count\n"
    "    first_sg = list(sg_map.values())[0]\n"
    "    patches = _generate_patch_regions(first_sg.n_rows, first_sg.n_cols, ph_patch, pw, 0, 0)\n"
    "    total_q = sum(len(sg.ion_indices) for sg in sg_map.values())\n"
    "    cached = sum(1 for p in plans_sc if p.identical_to_phase is not None)\n"
    "\n"
    "    rows.append({\n"
    '        "d": d_sc,\n'
    '        "qubits/block": len(first_sg.ion_indices),\n'
    '        "blocks": len(sg_map),\n'
    '        "grid": f"{first_sg.n_rows}\\u00d7{first_sg.n_cols}",\n'
    '        "traps/block": first_sg.n_rows * first_sg.n_cols,\n'
    '        "patches/tiling": len(patches),\n'
    '        "phases": len(plans_sc),\n'
    '        "cached": cached,\n'
    '        "total_ions": total_q,\n'
    "    })\n"
    "\n"
    "# Pretty-print table\n"
    'header = f"{\'d\':>3s}  {\'q/blk\':>5s}  {\'blks\':>4s}  {\'grid\':>6s}  {\'traps\':>5s}  {\'patch\':>5s}  {\'phase\':>5s}  {\'cache\':>5s}  {\'ions\':>5s}"\n'
    "print(header)\n"
    'print("-" * len(header))\n'
    "for r in rows:\n"
    '    print(f"{r[\'d\']:3d}  {r[\'qubits/block\']:5d}  {r[\'blocks\']:4d}  {r[\'grid\']:>6s}  "\n'
    '          f"{r[\'traps/block\']:5d}  {r[\'patches/tiling\']:5d}  {r[\'phases\']:5d}  "\n'
    '          f"{r[\'cached\']:5d}  {r[\'total_ions\']:5d}")\n'
    "\n"
    "print()\n"
    'print("Key insight: as d grows, Level 1 keeps blocks isolated and Level 2")\n'
    'print("keeps SAT patches small (\\u22644\\u00d73 traps). The number of patches per")\n'
    'print("tiling scales with grid area, but each patch is O(1) in complexity.")'
)


# ── Insert all new cells before Pipeline Reference ──
# Find Pipeline Reference index
pipeline_ref_idx = None
for i, c in enumerate(nb["cells"]):
    src = "".join(c["source"])
    if "Pipeline Reference" in src and c["cell_type"] == "markdown":
        pipeline_ref_idx = i
        break

if pipeline_ref_idx is None:
    print("ERROR: Pipeline Reference cell not found!")
    sys.exit(1)

print(f"Inserting new cells before Pipeline Reference at index {pipeline_ref_idx}")

new_cells = [
    md_phase_aware,
    code_step1,
    code_step2,
    code_step3,
    code_d3_run,
    md_d7_header,
    code_d7_partition,
    code_d7_patches,
    code_scaling,
]

for j, cell in enumerate(new_cells):
    nb["cells"].insert(pipeline_ref_idx + j, cell)
    src = "".join(cell["source"])[:50].replace("\n", " | ")
    print(f"  Inserted [{pipeline_ref_idx + j}] {cell['cell_type']:4s}: {src}...")

print(f"\nFinal cell count: {len(nb['cells'])}")

# Write
with open(NB_PATH, "w") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
    f.write("\n")

print("Done.")
