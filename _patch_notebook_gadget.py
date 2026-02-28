"""Patch trapped_ion_demo.ipynb to add gadget routing demo cells."""
import json, os

path = "notebooks/trapped_ion_demo.ipynb"
with open(path) as f:
    nb = json.load(f)

print(f"Before: {len(nb['cells'])} cells")

# ── Markdown header ──────────────────────────────────────────────
md_header = {
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "---\n",
        "## Phase-Aware Gadget Routing (Two-Level Spatial Slicing)\n",
        "\n",
        "End-to-end demonstration of the **gadget routing pipeline** from\n",
        "`GADGET_COMPILATION_SPEC.md`.\n",
        "\n",
        "The pipeline decomposes a fault-tolerant gadget experiment into temporal\n",
        "**phases** (EC rounds + gadget round), then routes each phase on the\n",
        "WISE grid using two levels of spatial slicing:\n",
        "\n",
        "| Level | Scope | Function |\n",
        "|-------|-------|----------|\n",
        "| **Level 1** | Block-level partitioning | `partition_grid_for_blocks()` assigns disjoint sub-grids per code block |\n",
        "| **Level 2** | Patch-and-route | `_patch_and_route()` SAT-based WISE routing within each sub-grid |\n",
        "\n",
        "**Key building blocks** exercised below:\n",
        "- `FaultTolerantGadgetExperiment` builds ideal circuit + `QECMetadata`\n",
        "- `decompose_into_phases()` produces `PhaseRoutingPlan` list\n",
        "- `route_full_experiment()` runs per-phase routing with EC round caching\n",
        "- `compute_schedule_timing()` computes analytical execution & reconfiguration time",
    ],
}


def _make_code_cell(source_str):
    lines = source_str.split("\n")
    source = [l + "\n" for l in lines[:-1]] + [lines[-1]]
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source,
    }


# ── Step 1: Build experiment ─────────────────────────────────────
step1 = _make_code_cell(
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

# ── Step 2: Level 1 partition ────────────────────────────────────
step2 = _make_code_cell(
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
    '    print(f"  [{i}] type={p.phase_type:8s}  blocks={p.block_names}  "\n'
    '          f"ms_rounds={rounds_info}  cacheable={p.cache_key is not None}")'
)

# ── Step 3: Full routing ─────────────────────────────────────────
step3 = _make_code_cell(
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

# Insert: after cell 13 (noisy simulation code), before cell 14 (pipeline ref)
insert_pos = 14
nb["cells"] = nb["cells"][:insert_pos] + [md_header, step1, step2, step3] + nb["cells"][insert_pos:]

tmp = path + ".tmp"
with open(tmp, "w") as f:
    json.dump(nb, f, indent=1)
os.replace(tmp, path)

print(f"After: {len(nb['cells'])} cells")
print("Done.")
