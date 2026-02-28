# CSS Surgery Animation — Fix Plan

> Comprehensive diagnosis and fix plan for the issues visible in the CSS
> surgery gadget animation in `trapped_ion_demo.ipynb`.

---

## Issue Summary

| # | Symptom | Severity | Root Cause |
|---|---------|----------|------------|
| 1 | Stim timeslice shows blocks overlapping / as lines, not 2-D patches | **HIGH** | Block coordinate offsets produce overlapping or degenerate bounding boxes |
| 2 | Qubits crammed to the left, huge whitespace on the right | **HIGH** | Bridge ancillas or fallback coords inflate the viewBox far to the right |
| 3 | Random qubits appear far right in the timeslice | **HIGH** | Same — outlier QUBIT_COORDS from bridge ancillas or coord fallbacks |
| 4 | Bridge ancillas not positioned at block boundaries | **HIGH** | `add_boundary_bridges` midpoint logic doesn't account for xy offsets correctly for the L-shaped CSS surgery layout |
| 5 | Initial ion mapping mangles all blocks together | **CRITICAL** | `WISEArchitecture()` created without `compact_clustering=True` / `add_spectators=True` in `compile_gadget_for_animation`; or metadata not injected early enough for `_map_qubits_per_block` dispatch |
| 6 | Multiple global reconfigurations before first MS round | **MEDIUM** | Per-block schedules are concatenated rather than merged; transition steps are separate per block |
| 7 | RX/RY rotations not optimally batched between MS rounds | **LOW** | `_drain_single_qubit_ops` already sorts by type, but rotation window boundaries could be tighter |
| 8 | Resets not batched across consecutive reset rounds | **MEDIUM** | Epoch boundaries prevent merging resets from distinct stim TICKs |
| 9 | Pre-gadget EC rounds metadata bug | **MEDIUM** | `QECMetadata` always sets `active_blocks=ALL` for `stabilizer_round_pre`, ignoring `get_blocks_to_skip_pre_rounds()` |
| 10 | Highlighted stim instructions don't match current animation step | **MEDIUM** | `_stim_origin` mapping breaks for CSS surgery due to interleaved EC+merge phases |

---

## Fix 1 — CSS Surgery Block Layout Coordinates (Issues 1, 2, 3, 4)

### Root Cause

`CSSSurgeryCNOTGadget.compute_layout()` at
[css_surgery_cnot.py:1195](src/qectostim/gadgets/css_surgery_cnot.py#L1195)
places 3 blocks in an **L-shape**:

```
block_0 (ctrl)  ──gap──  block_1 (ancilla)
                                |
                               gap
                                |
                          block_2 (target)
```

- block_0 at `(0, 0)`
- block_1: left edge at block_0's right edge + gap=2
- block_2: top edge at block_1's bottom edge + gap=2, X-center aligned with block_1

For d=2 rotated surface code, qubit coords span `[0, 4] × [0, 4]`
(ancillas at 0 and 4, data at 1 and 3). The bounding box is 4×4 per block.

The resulting **global coordinates** are approximately:

| Block | X range | Y range |
|-------|---------|---------|
| block_0 | [0, 4] | [0, 4] |
| block_1 | [6, 10] | [0, 4] |
| block_2 | [6, 10] | [-6, -2] |

**Bridge ancillas** (ZZ merge between block_0 and block_1) are at midpoints
of the right boundary of block_0 and left boundary of block_1, so
approximately `x ≈ 5, y ∈ {1, 3}`. XX merge bridges (between block_1 and
block_2) are at `x ∈ {7, 9}, y ≈ -1`.

**The coordinates themselves are correct** — the issue is that:

1. **stim's timeslice SVG** takes the union viewBox across all ticks. If
   some ticks only involve block_0 qubits (`x ∈ [0,4]`) but the global
   viewBox spans `[0, 10] × [-6, 4]` (a 10×10 area), those qubits appear
   tiny and crammed into one corner.

2. **The L-shape wastes ~50% of the viewBox** area (upper-left quadrant
   has block_0, upper-right has block_1, lower-right has block_2, lower-left
   is empty). Combined with bridge ancillas at extreme positions, this
   "zooms out" the view.

### Fix

**A. Tighter block arrangement** — Change `compute_layout()` to stack
blocks **vertically in a column** instead of the L-shape:

```
block_0 (ctrl)      offset = (0, 0)
──── gap ────
block_1 (ancilla)   offset = (0, -(block_height + gap))
──── gap ────
block_2 (target)    offset = (0, -(2 * (block_height + gap)))
```

This produces a tall, narrow bounding box instead of an L, reducing
viewBox waste from ~50% to ~0%.

ZZ merge bridges (block_0 ↔ block_1): placed along the shared
**bottom/top** boundary between block_0 and block_1. XX merge bridges
(block_1 ↔ block_2): placed along the shared **bottom/top** boundary
between block_1 and block_2.

**Alternative**: Keep the L-shape but change the animation's `draw_stim_svg`
to **pad viewBox to square** and use **per-block zoom insets**.

**B. Normalise timeslice viewBox per-tick** (optional) — Instead of
unioning all tick viewBoxes, compute viewBox from only the **active
qubits** in each tick. This prevents inactive blocks from inflating the
view.

### Files to Change

| File | Function | Change |
|------|----------|--------|
| `src/qectostim/gadgets/css_surgery_cnot.py` | `compute_layout()` | Stack blocks vertically (or at least in a rectangle, not L) |
| `src/qectostim/gadgets/css_surgery_cnot.py` | `add_boundary_bridges()` calls | Verify bridge coords interpolate correctly for new layout |
| `src/qectostim/experiments/hardware_simulation/trapped_ion/viz/animation.py` | viewBox unioning (~L230) | Option: use per-tick viewBox or add padding/centering logic |

---

## Fix 2 — Initial Ion Mapping: Per-Block Spatial Partitioning (Issue 5)

### Root Cause

`compile_gadget_for_animation()` at
[run.py:479](src/qectostim/experiments/hardware_simulation/trapped_ion/demo/run.py#L479)
creates:

```python
arch = WISEArchitecture(wise_config=wise_cfg)  # ← missing compact_clustering, add_spectators
```

Without `compact_clustering=True` and `add_spectators=True`, the
`build_topology_per_block()` call may not produce the same spatial
clustering behaviour as the regular compiler path.

Additionally, the `_use_per_block` guard in `compile_gadget_for_animation()`
checks for `≥2 block_allocations`, which is satisfied for CSS surgery (3
blocks). However, if `qubit_allocation` is `None` (caller forgot to pass
it), the fallback path runs **global** `build_topology()` and all ions
get clustered together across the entire grid.

### Fix

**A.** Pass `compact_clustering=True, add_spectators=True` to
`WISEArchitecture()` constructor in `compile_gadget_for_animation()`:

```python
arch = WISEArchitecture(
    wise_config=wise_cfg,
    add_spectators=True,
    compact_clustering=True,
)
```

**B.** Add a loud warning (or raise) if `qec_metadata` is provided but
`qubit_allocation` is `None` — this silently falls through to a broken
global path.

**C.** Ensure `allocate_block_regions()` at
[gadget_routing.py:310](src/qectostim/experiments/hardware_simulation/trapped_ion/utils/gadget_routing.py#L310)
lays out blocks to **match the stim coordinate layout** — i.e., if the
stim circuit has block_0 at the top and block_2 at the bottom, the grid
regions should mirror that vertical arrangement. Currently it lays all
blocks **side-by-side along columns**, which works but doesn't match the
stim visual layout (especially for the CSS surgery L-shape). After Fix 1
(vertical stacking), the grid partitioning should also stack blocks
vertically.

### Files to Change

| File | Function | Change |
|------|----------|--------|
| `demo/run.py` | `compile_gadget_for_animation()` | Add `compact_clustering=True, add_spectators=True` |
| `demo/run.py` | `compile_gadget_for_animation()` | Add warning if `qubit_allocation is None` with `qec_metadata` |
| `gadget_routing.py` | `allocate_block_regions()` | Option: match grid layout to stim coordinate layout direction |

---

## Fix 3 — Reconfiguration Merging Across Disjoint Blocks (Issue 6)

### Root Cause

`_merge_disjoint_block_schedules()` at
[qccd_WISE_ion_route.py:2343](src/qectostim/experiments/hardware_simulation/trapped_ion/compiler/qccd_WISE_ion_route.py#L2343)
already merges per-block SAT schedules for the **same MS round**. However,
**transition reconfigurations** (between phases) are computed per-block
and generate separate RoutingStep objects. The execution loop at
[qccd_WISE_ion_route.py:2200](src/qectostim/experiments/hardware_simulation/trapped_ion/compiler/qccd_WISE_ion_route.py#L2200)
applies each reconfiguration step sequentially, producing multiple
`GlobalReconfigurations` batches where a single merged one would suffice.

The problem: `_compute_transition_reconfig_steps()` at
[gadget_routing.py:1402](src/qectostim/experiments/hardware_simulation/trapped_ion/utils/gadget_routing.py#L1402)
generates **per-block** transition steps. When these are interleaved with
normal routing steps and have no `solved_pairs`, the execution loop
produces one reconfiguration per step.

### Fix

**A.** In `_compute_transition_reconfig_steps()`, merge all per-block
transition schedules into a **single** RoutingStep using
`_merge_disjoint_block_schedules()` before returning. This already exists
for intra-round merging but isn't applied to transitions.

**B.** In the execution loop, detect sequences of consecutive
reconfiguration-only steps (empty `solved_pairs`) and merge their
schedules before applying as a single `GlobalReconfigurations` operation.

### Files to Change

| File | Function | Change |
|------|----------|--------|
| `gadget_routing.py` | `_compute_transition_reconfig_steps()` | Merge per-block transition schedules into 1 RoutingStep |
| `qccd_WISE_ion_route.py` | execution loop (~L2200) | Optionally: coalesce adjacent reconfig-only steps |

---

## Fix 4 — Rotation and Reset Batching (Issues 7, 8)

### Root Cause

`_drain_single_qubit_ops()` at
[qccd_WISE_ion_route.py:2000](src/qectostim/experiments/hardware_simulation/trapped_ion/compiler/qccd_WISE_ion_route.py#L2000)
already batches by type within each epoch. The **_TYPE_ORDER** is:

```python
{XRotation: 0, YRotation: 1, QubitReset: 2, Measurement: 3}
```

This drains all XRotation first, then YRotation, then resets, then
measurements — **within a single epoch**. The issue is:

1. **Cross-epoch batching**: Resets from epoch `e` and resets from epoch
   `e+1` on **disjoint qubits** are not merged because `_drain` processes
   epochs sequentially with hard boundaries. For example, if epoch 0 resets
   qubits {0,1,2} and epoch 1 resets qubits {3,4,5}, they generate two
   separate batches instead of one.

2. **Rotation postponement**: If qubit Q has `RX` in epoch 0 followed by
   a reset+`RX` in epoch 1 (X-basis reset = R + RX), the epoch 0 `RX`
   and epoch 1 `RX` could be merged if the intervening reset is on a
   different qubit. But epoch boundaries prevent this.

### Fix

**A. Cross-epoch reset merging**: After the type-priority drain within
each epoch, check if the **next epoch** starts with the same gate type
on disjoint qubits. If so, merge them into the same parallel batch.

Add a post-processing pass:

```python
def _merge_adjacent_same_type_batches(batches):
    """Merge consecutive batches of the same gate type on disjoint qubits."""
    merged = []
    for batch in batches:
        if merged and merged[-1].type == batch.type and \
           merged[-1].ions.isdisjoint(batch.ions):
            merged[-1] = merge(merged[-1], batch)
        else:
            merged.append(batch)
    return merged
```

**B.** `reorder_rotations_for_batching()` at
[qccd_parallelisation.py:66](src/qectostim/experiments/hardware_simulation/trapped_ion/compiler/qccd_parallelisation.py#L66)
exists but is **not called in production**. Wire it into the pipeline at
the appropriate point (after decompose, before scheduling) or integrate
its logic into `_drain_single_qubit_ops`.

**C. Rotation postponement**: In `_drain_single_qubit_ops`, when building
per-ion queues, look ahead across epoch boundaries for same-type rotations
on disjoint qubits and pull them into the current drain window. Constrained
by: never pull past a multi-qubit gate on the same ion.

### Files to Change

| File | Function | Change |
|------|----------|--------|
| `qccd_WISE_ion_route.py` | `_drain_single_qubit_ops()` | Cross-epoch merging of same-type ops on disjoint qubits |
| `qccd_parallelisation.py` | `reorder_rotations_for_batching()` | Optionally integrate into production pipeline |
| `qccd_WISE_ion_route.py` | execution loop | Post-processing merge of adjacent same-type batches |

---

## Fix 5 — Pre-Gadget EC Rounds Metadata (Issue 9)

### Root Cause

`QECMetadata.from_gadget_experiment()` at
[pipeline.py:555](src/qectostim/experiments/hardware_simulation/core/pipeline.py#L555)
always sets `stabilizer_round_pre.active_blocks = all_block_names`,
ignoring `gadget.get_blocks_to_skip_pre_rounds()`.

For CSS surgery (flat), `get_blocks_to_skip_pre_rounds()` returns
`{"block_0", "block_1", "block_2"}` — i.e., ALL blocks are skipped.
But the metadata claims pre-EC runs on all 3 blocks, causing the router
to generate routing plans for EC rounds that don't exist in the circuit.

### Fix

```python
# In pipeline.py, from_gadget_experiment():
skipped_pre = gadget.get_blocks_to_skip_pre_rounds()
pre_active = [b for b in all_block_names if b not in skipped_pre]

PhaseInfo(
    phase_type="stabilizer_round_pre",
    num_rounds=rounds_before,
    active_blocks=pre_active,  # ← was: all_block_names
)
```

Same for `stabilizer_round_post`:

```python
skipped_post = gadget.get_blocks_to_skip_post_rounds()
post_active = [b for b in all_block_names if b not in skipped_post]
```

### Files to Change

| File | Function | Change |
|------|----------|--------|
| `src/qectostim/experiments/hardware_simulation/core/pipeline.py` | `from_gadget_experiment()` | Consult `get_blocks_to_skip_pre_rounds()` and `get_blocks_to_skip_post_rounds()` for active_blocks |

---

## Fix 6 — Stim Instruction Highlighting (Issue 10)

### Root Cause

The `_stim_origin` attribute is set during `decompose_to_native()` at
[trapped_ion_compiler.py:177](src/qectostim/experiments/hardware_simulation/trapped_ion/utils/trapped_ion_compiler.py#L177).
Each native operation gets tagged with the index of its source stim gate
in the sidebar entry list.

For CSS surgery, the gadget generates **interleaved EC + merge rounds**
with complex phase structure. The sidebar entry list is built by
`parse_stim_for_sidebar()` which processes the **ideal** stim circuit
linearly. But the native decomposition reorders operations (e.g., all
R gates before all H gates before CX gates). The `_stim_origin` index
may point to the wrong sidebar entry when the ideal circuit has
non-standard ordering (merge phases interleave CX gates from different
EC rounds).

Additionally, stim's `Circuit.decomposed()` merges adjacent same-type
gates (e.g., `R 6 + R 11 12 → R 6 11 12`), causing offset drift in the
`_decomp_to_sidebar` mapping.

### Fix

**A.** Use per-gate decomposition (Step 3) instead of whole-circuit
`decomposed()` when building `_decomp_to_sidebar` — this is already
implemented per the stored memory fact but may need verification for CSS
surgery circuits.

**B.** Validate `_stim_origin` values in `_apply_authoritative_mapping()`
by cross-checking the gate kind (MS origin should point to a CX/CZ
sidebar entry, rotation origin should point to an H/S sidebar entry, etc.).
Add a fallback to qubit-overlap matching when the cross-check fails.

### Files to Change

| File | Function | Change |
|------|----------|--------|
| `trapped_ion_compiler.py` | `decompose_to_native()` | Verify per-gate decomp used for `_stim_origin` |
| `viz/animation.py` | `_apply_authoritative_mapping()` | Add gate-kind cross-check for `_stim_origin` values |

---

## Fix 7 — First CX Round Not Nearest-Neighbour

### Root Cause

The first round of CX gates in the CSS surgery circuit involves
**bridge ancillas** connecting blocks. Bridge ancilla qubit indices are
assigned at the end of the global index space (after all block qubits).
If the initial ion mapping doesn't place bridge ancilla ions near the
boundary of the relevant blocks, the first MS round requires long-range
transport.

With Fix 2 (per-block spatial partitioning), bridge ancillas are
assigned to one of the connected blocks and placed in that block's
sub-grid. However, `regularPartition()` doesn't know about the special
role of bridge ancillas — it clusters them purely by spatial proximity
to data/measurement ions.

### Fix

**A.** In `build_topology_per_block()`, when building measurement vs data
ion lists per block, flag bridge ancilla ions as **boundary ions** and
ensure `regularPartition()` places them in traps at the **edge** of the
block's sub-grid closest to the adjacent block.

**B.** Alternative: In `allocate_block_regions()`, when bridge ancillas
are assigned to a block, place them in a thin **boundary strip** between
the two blocks rather than inside either block. This requires a dedicated
"bridge zone" in the grid (1-column wide, shared between blocks).

### Files to Change

| File | Function | Change |
|------|----------|--------|
| `architectures.py` | `build_topology_per_block()` | Bias bridge ancilla ions toward block boundary traps |
| `gadget_routing.py` | `allocate_block_regions()` | Optional: add boundary strip for bridge ions |

---

## Implementation Priority

| Priority | Fix | Effort | Impact |
|----------|-----|--------|--------|
| **P0** | Fix 2 — Per-block spatial partitioning (`compact_clustering=True`) | Small | Fixes mangled initial mapping |
| **P0** | Fix 5 — Pre-gadget EC metadata | Small | Prevents phantom routing plans |
| **P1** | Fix 1 — Block layout coordinates | Medium | Fixes timeslice visualization |
| **P1** | Fix 3 — Reconfiguration merging | Medium | Reduces redundant reconfig steps |
| **P2** | Fix 6 — Stim highlighting | Medium | Cosmetic — animation accuracy |
| **P2** | Fix 4 — Rotation/reset batching | Medium | Performance optimization |
| **P3** | Fix 7 — Bridge ancilla placement | Hard | Reduces first-round transport |

---

## Testing

### Quick validation (no SAT — metadata/layout only)

```bash
cd /path/to/QECToStim
source my_venv/bin/activate
PYTHONPATH=src python -c "
from qectostim.gadgets.css_surgery_cnot import CSSSurgeryCNOTGadget
from qectostim.codes.surface.rotated_surface import RotatedSurfaceCode
from qectostim.experiments.ft_gadget_experiment import FaultTolerantGadgetExperiment

g = CSSSurgeryCNOTGadget()
code = RotatedSurfaceCode(distance=2)
ft = FaultTolerantGadgetExperiment(codes=[code], gadget=g, noise_model=None,
                                    num_rounds_before=2, num_rounds_after=2)
circ = ft.to_stim()
meta = ft.qec_metadata
alloc = ft._unified_allocation

# Check block coordinates don't overlap
for ba in meta.block_allocations:
    print(f'{ba.block_name}: offset={ba.offset}, data={list(ba.data_qubits)[:3]}...')

# Check pre-gadget EC active blocks
for ph in meta.phases:
    if ph.phase_type == 'stabilizer_round_pre':
        print(f'Pre-EC active_blocks: {ph.active_blocks}')

# Check bridge ancilla coords
for gi, coord, purpose in alloc.bridge_ancillas:
    print(f'Bridge {gi}: coord={coord}, purpose={purpose}')
"
```

### Full animation validation (with SAT)

```bash
WISE_INPROCESS_LIMIT=999999999 PYTHONPATH=src my_venv/bin/python -c "
# Run the CSS surgery cell from the notebook and check:
# 1. Blocks appear as 2-D patches, not lines
# 2. No huge whitespace in timeslice
# 3. Bridge ancillas at block boundaries
# 4. Ion mapping shows per-block clustering
# 5. Highlighted instructions match current step
"
```

### Unit tests

```bash
PYTHONPATH=src python -m pytest src/qectostim/experiments/hardware_simulation/trapped_ion/demo/test_e2e.py -x -q --tb=short
```
