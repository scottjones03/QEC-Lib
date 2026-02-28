# CSS Surgery Animation — Comprehensive Fix Plan v2

> Updated diagnosis and fix plan consolidating all reported issues in the
> CSS surgery gadget animation (`trapped_ion_demo.ipynb`, cell 24).
>
> **Supersedes**: `CSS_SURGERY_ANIMATION_FIX_PLAN.md` (partial), prior
> session fixes (bridge double-counting, `_execute_ms_gates` scan,
> `active_blocks=[]` fallback, interleaved EC).

---

## Already Fixed (Prior Sessions)

| Fix | File | Status |
|-----|------|--------|
| Bridge double-counting in `allocate_qubits()` | `layout.py:1059` | ✅ Verified |
| `_execute_ms_gates` scan-all-remaining approach | `qccd_WISE_ion_route.py:~2155` | ✅ Unit-tested |
| `active_blocks=[]` treated as falsy → fixed to `is not None` | `gadget_routing.py:~968,~1543` | ✅ Unit-tested |
| Interleaved EC rounds in gadget phases | `gadget_routing.py:~1084` | ✅ Verified (10 rounds per merge) |
| Pre/post EC `active_blocks` from `get_blocks_to_skip_{pre,post}_rounds()` | `pipeline.py:~568` | ✅ Already applied |

---

## Open Issues — Numbered Reference

| # | Symptom | Severity | Category |
|---|---------|----------|----------|
| **A** | Stim timeslice: blocks in one long line, not L-shaped/columnar | HIGH | Layout / Visual |
| **B** | Bridge ancillas as random qubits far to the right | HIGH | Layout / Visual |
| **C** | Non-local CX gates in merge phase timeslice | HIGH | Routing / Layout |
| **D** | First MS round ≠ first CX round highlighting | HIGH | Highlighting |
| **E** | Missing RX rotations — only 1 before first MS round | HIGH | Compiler |
| **F** | Merge/split MS rounds not scheduled in parallel | MEDIUM | Scheduling |
| **G** | Final measurements not compiled (instructions dropped) | HIGH | Compiler / Execution |
| **H** | Multiple global reconfigurations before first CX | MEDIUM | Routing |
| **I** | Initial mapping mangles all blocks together | CRITICAL | Grid Partitioning |
| **J** | Poor batching of resets/rotations between MS rounds | LOW | Scheduling |
| **K** | First CX round not nearest-neighbour in block | MEDIUM | Clustering |
| **L** | Missing d EC rounds on all blocks before gadget | MEDIUM | Metadata |
| **M** | Global reconfigs not merged between disjoint blocks | LOW | Routing |

---

## Root Cause Analysis

### Issue A + B — Stim Timeslice Layout

**Root cause**: The stim circuit's `QUBIT_COORDS` stack blocks **vertically**
(block_0 at y∈[0,4], block_1 at y∈[-6,-2], block_2 at y∈[-12,-8]) while
bridge ancillas (qubit 21) are at y=-1 and (qubits 22,23) are at y=-7.
This is correct for a column layout. However:

1. stim's `diagram('timeslice-svg')` computes the **global viewBox** from
   the union of ALL qubit coords. Many ticks only involve a subset of
   blocks, making those qubits appear tiny in the zoomed-out view.

2. The gap between block_0 (y=0..4) and block_2 (y=-12..-8) is 16 units
   vertically but only 4 units horizontally → extremely tall thin SVG.
   The animation panel's aspect ratio squishes this.

3. Bridge qubit 21 at (1,-1) appears between block_0 and block_1 but
   qubits 22,23 at (1,-7) and (3,-7) are in the gap between block_1
   and block_2 — correct positioning but looks like "random qubits" when
   zoomed out.

**Fix A1 — Per-tick adaptive viewBox** (animation.py):
Instead of a single global viewBox for all ticks, compute the viewBox
from only the **active qubits** (those touched by gates) in each tick,
with padding to keep the view stable. When all blocks are active (merge
phases), use the full viewBox; when only one block is active (EC rounds),
zoom to that block's bounding box.

```python
# In _build_stim_panel() or parse_stim_timeslice_svg():
def _compute_adaptive_viewbox(active_qubit_coords, padding=2.0):
    xs = [c[0] for c in active_qubit_coords]
    ys = [c[1] for c in active_qubit_coords]
    return (min(xs)-padding, min(ys)-padding,
            max(xs)-min(xs)+2*padding, max(ys)-min(ys)+2*padding)
```

**Fix A2 — Square aspect ratio padding** (stim_panel.py):
Pad the SVG viewBox to a square to prevent extreme aspect ratio distortion.
Add proportional margins.

**Files**: `viz/stim_panel.py`, `viz/animation.py`

---

### Issue C — Non-Local CX in Merge Phase

**Root cause**: The stim circuit's merge CX gates involve bridge ancillas
connecting data qubits from **different blocks**. For example, ZZ merge
round 1:

```
CX 0 21 1 21 2 21 3 21    ← block_0 data ↔ bridge qubit 21
CX 7 21 8 21 9 21 10 21   ← block_1 data ↔ bridge qubit 21
```

In the stim timeslice, these appear as CX connecting qubits with very
different coordinates (block_0 data at y∈[1,3] and block_1 data at
y∈[-5,-3]), producing long lines. This is **correct by design** — CSS
surgery merge phases inherently involve non-local CX between blocks via
bridge ancillas.

The timeslice will always show long-range CX during merge — this is not
a bug. Fix A1/A2 (tighter viewBox) will make these connections more
readable.

**No code change needed** — just the layout fix above.

---

### Issue D — First MS Round ≠ First CX Highlighted

**Root cause**: Two contributing factors:

1. **Routing phase order ≠ stim CX order**: The routing engine may
   execute MS rounds in a different order than the stim circuit. The
   `_execute_ms_gates()` scan-all-remaining fix (already applied) helps
   match any pair, but the **highlighting** still depends on which pairs
   are in the first routing step's `solved_pairs` — these may not correspond
   to the first CX layer in the stim circuit.

2. **`_stim_origin` offset drift**: When `decompose_to_native()` processes
   multi-qubit gates (e.g., `RX 7 8 9 10` is one stim instruction but
   produces 3 native ops × 4 qubits = 12 ops), the `_origin` index
   advances once per stim instruction. But stim's `Circuit.decomposed()`
   can merge adjacent same-type gates, causing the sidebar entry count to
   diverge from the per-gate `_origin` count.

**Fix D1 — Stim-order MS execution** (qccd_WISE_ion_route.py):
Instead of scanning all remaining ops in `_execute_ms_gates()`, maintain
a **stim CX index** that tracks which CX instruction layer we're executing.
The first MS round should try to execute pairs matching the first CX
layer's qubit pairs in the stim circuit.

```python
# Build stim_cx_order from the stim circuit at compile time
stim_cx_layers = []  # [(qubit_pair_set, stim_instruction_index), ...]
for inst in stim_circuit:
    if inst.name in ("CX", "CNOT"):
        pairs = set()
        targets = inst.targets_copy()
        for i in range(0, len(targets), 2):
            pairs.add(frozenset({targets[i].value, targets[i+1].value}))
        stim_cx_layers.append(pairs)
```

Then in `_execute_ms_gates()`, prefer matching against `stim_cx_layers[cx_idx]`
first, falling back to scan-all if no match found.

**Fix D2 — Per-gate sidebar mapping** (trapped_ion_compiler.py):
Verify that `_stim_origin` is set using per-gate (not whole-circuit)
decomposition indices. This was previously identified as the
`decomposed()` merging issue — ensure the fix is still in place for CSS
surgery circuits.

**Files**: `qccd_WISE_ion_route.py`, `trapped_ion_compiler.py`

---

### Issue E — Missing RX Rotations

**Root cause**: The stim circuit contains `RX` (X-basis reset = prepare
|+⟩) which the compiler routes through `decompose_stim_gate()`:

```
RX → R (QubitReset) + RY(π/2) (YRotation) + RX(π/2) (XRotation)
```

This produces **3 native ops** per qubit. The user sees "only 1 RX before
first MS" — but this may be correct if:
- The `QubitReset` ops are batched in a separate earlier group (resets first)
- The `YRotation` is in the next group
- Only the `XRotation` appears right before MS

This is actually the expected behaviour of the type-priority drain:
`_TYPE_ORDER = {XRotation: 0, YRotation: 1, QubitReset: 2, Measurement: 3}`.
XRotation drains first (appears right before MS), then YRotation, then
Reset. The ordering is **backwards from physical execution order** — reset
should come FIRST, then rotation.

**Fix E1 — Correct `_TYPE_ORDER`** (qccd_WISE_ion_route.py):
The type order should prioritize reset before rotations:

```python
_TYPE_ORDER = {QubitReset: 0, XRotation: 1, YRotation: 2, Measurement: 3}
```

This ensures QubitReset executes first (preparing |0⟩), then XRotation
and YRotation (rotating to the target basis).

**Fix E2 — Handle `RX`, `MRX`, `MX` directly in decompose_to_native()**
(trapped_ion_compiler.py):
Instead of falling through to `decompose_stim_gate()`, add explicit
handling:

```python
elif gate_name == "RX":
    # X-basis reset: R → H (= R → RY → RX)
    for qi in qubit_parts:
        idx = int(qi)
        ion = self._ion_mapping[idx][0]
        operations.append(_tag(QubitReset.qubitOperation(ion), _origin, _epoch))
        operations.append(_tag(YRotation.qubitOperation(ion), _origin, _epoch))
        operations.append(_tag(XRotation.qubitOperation(ion), _origin, _epoch))

elif gate_name == "MX":
    # X-basis measurement: H → M (= RX → RY → M)
    for qi in qubit_parts:
        idx = int(qi)
        ion = self._ion_mapping[idx][0]
        operations.append(_tag(XRotation.qubitOperation(ion), _origin, _epoch))
        operations.append(_tag(YRotation.qubitOperation(ion), _origin, _epoch))
        operations.append(_tag(Measurement.qubitOperation(ion), _origin, _epoch))

elif gate_name == "MRX":
    # X-basis measure + X-basis reset: MX → R → H
    for qi in qubit_parts:
        idx = int(qi)
        ion = self._ion_mapping[idx][0]
        operations.append(_tag(XRotation.qubitOperation(ion), _origin, _epoch))
        operations.append(_tag(YRotation.qubitOperation(ion), _origin, _epoch))
        operations.append(_tag(Measurement.qubitOperation(ion), _origin, _epoch))
        operations.append(_tag(QubitReset.qubitOperation(ion), _origin, _epoch))
        operations.append(_tag(YRotation.qubitOperation(ion), _origin, _epoch))
        operations.append(_tag(XRotation.qubitOperation(ion), _origin, _epoch))
```

Note: This handles multi-qubit `RX 4 5` (line 36 in stim circuit) by
iterating over all qubit parts. The current fallback path only processes
`qubit_parts[0]` (first qubit), **dropping all subsequent qubits**.
This is the actual root cause of "missing RX" — multi-qubit `RX`/`MRX`/`MX`
instructions lose all but the first qubit.

**Fix E3 — Multi-qubit fallback processing** (trapped_ion_compiler.py):
The main decompose loop at line ~410 does:
```python
idx = int(qubit_parts[0])
ion = self._ion_mapping[idx][0]
```
This only processes the **first** qubit of multi-qubit instructions like
`RX 4 5` or `MRX 11 12`. The fallback at line ~469 passes ALL qubit
indices to `decompose_stim_gate()`, which correctly returns per-qubit ops.
But the early `idx = int(qubit_parts[0])` extraction at the top of the
loop body means multi-qubit `M`, `R`, `MR` are also only processing the
first qubit.

Verify: `R 0 1 2 3 4 5 6 ...` (line 24) — does this reset all qubits?
The `R` handler at line ~432 only uses `ion` (first qubit). This is a
**global bug** — ALL multi-qubit single-target instructions only process
the first qubit.

**Critical Fix**: Restructure the decompose loop to iterate over ALL
qubit parts for single-qubit instructions:

```python
if gate_name in ("M", "MZ"):
    for qi in qubit_parts:
        idx = int(qi)
        ion = self._ion_mapping[idx][0]
        operations.append(_tag(Measurement.qubitOperation(ion), _origin, _epoch))
elif gate_name == "R":
    for qi in qubit_parts:
        idx = int(qi)
        ion = self._ion_mapping[idx][0]
        operations.append(_tag(QubitReset.qubitOperation(ion), _origin, _epoch))
# ... etc for H, RX, MX, MRX, MR
```

**Files**: `trapped_ion_compiler.py` (lines ~410-530), `qccd_WISE_ion_route.py` (`_TYPE_ORDER`)

### SEVERITY: This is the **most critical bug**. Multi-qubit R/M/RX/MX instructions are losing all but the first qubit, which means most qubits never get reset, measured, or rotated. This explains:
- "Only 1 RX before first MS" (only first qubit of `RX 4 5` processed)
- "Final measurements missing" (`M 0 1 2 3 14 15 16 17` only measures qubit 0)
- "Dropping rotations" throughout

---

### Issue F — Merge/Split MS Rounds Not Parallelized

**Root cause**: Each merge round's bridge CX pairs connect bridge ancilla
to data qubits from TWO blocks. These pairs share the bridge ancilla ion,
so they **cannot** execute in parallel (same ion). Within a single merge
round, parallelism is limited to bridge CX on **different** bridge ancillas.

For CSS surgery d=2: each merge round has only 1 bridge ancilla (1 for ZZ,
2 for XX), so at most 2 MS pairs can execute in parallel per round. The
SAT solver's `solved_pairs` already contains pairs that can run in parallel
within one routing step. If the solution only has 1 pair per step, that's
the routing solver not finding parallelism.

**Fix F1** — Verify `_merge_phase_pairs()` correctly identifies
disjoint-ion pairs for parallel execution. For merge phases, pairs sharing
the bridge ancilla cannot merge. This is already handled correctly.

**Fix F2** — For EC rounds interleaved within merge phases: EC CX pairs
are block-internal and on different ions from bridge CX. EC rounds from
**different blocks** can run in parallel. The interleaving code already
merges per-block EC pairs into single rounds. Verify the SAT solver
receives these merged rounds.

**Files**: `gadget_routing.py` (`_merge_phase_pairs`), No change likely needed.

---

### Issue G — Final Measurements Not Compiled

**Root cause**: Directly caused by **Issue E** — the multi-qubit `M`
instruction `M 0 1 2 3 14 15 16 17` (line 241 of stim circuit) only
processes qubit 0. Qubits 1-3, 14-17 are never measured.

Additionally, `MX 22 23` (line 189) and `MX 7 8 9 10` (line 189) go
through the `decompose_stim_gate` fallback which correctly handles
multi-qubit, but only because they hit the fallback path. `M` has an
explicit handler that only processes the first qubit.

**Fix**: Same as Fix E3 (multi-qubit loop). Once all qubits are processed,
the final drain will find and execute all remaining measurements.

**Files**: Same as Issue E.

---

### Issue H — Multiple Reconfigurations Before First CX

**Root cause**: The routing generates RoutingSteps for:
1. **Initial placement** (`is_initial_placement=True`) — places ions on grid
2. **EC phase routing** — SAT-solves each EC round, potentially producing
   multiple reconfig steps per round
3. **Transition between EC phase and gadget phase** — `_compute_transition_reconfig_steps()`
   generates layout transition steps

For CSS surgery (flat codes), pre-gadget EC is skipped (`active_blocks=[]`
after the fix). So the routing should go directly from initial placement
to the first gadget phase. The "multiple reconfigs" may come from:
- Multiple routing steps within the first gadget round (SAT solver
  generates multi-step solutions for complex pair sets)
- Transition reconfigs that should be merged

**Fix H1 — Coalesce adjacent reconfig-only steps** (qccd_WISE_ion_route.py):
In the execution loop, when consecutive RoutingSteps have empty
`solved_pairs` (reconfig-only), merge their schedules and apply as one
`GlobalReconfigurations` operation.

```python
# Before applying reconfig schedule, look ahead for more reconfig-only steps
while (step_idx + 1 < len(round_steps)
       and not round_steps[step_idx + 1].solved_pairs):
    step_idx += 1
    merged_schedule.extend(round_steps[step_idx].schedule)
```

**Files**: `qccd_WISE_ion_route.py` (execution loop ~L2200)

---

### Issue I — Initial Mapping Mangles All Blocks Together

**Root cause**: `allocate_block_regions()` in `gadget_routing.py` lays
blocks **side-by-side along columns** at row=0. This creates disjoint
column regions. Then `build_topology_per_block()` runs `regularPartition`
+ `hillClimb` within each block's sub-grid independently.

The issue is that `allocate_block_regions()` uses `ceil(sqrt(n_qubits))`
for row count and places all blocks at `row=0` with incrementing column
offsets. For CSS surgery d=2 with 3 blocks:
- block_0: 9 qubits → 3×2 region at cols [0,2)
- block_1: 9 qubits → 3×2 region at cols [2,4)
- block_2: 9 qubits → 3×2 region at cols [4,6)

This is **correct for routing** but doesn't match the stim layout (which
stacks vertically). The routing grid is purely functional — it doesn't
need to match the stim coordinate layout.

The "mangled together" issue likely comes from the **animation display**
which overlays the ion positions on the grid. If the grid layout is 3×6
(very wide) but the stim SVG shows a column, they look disconnected.

**Fix I1 — Match grid allocation direction to stim layout**
(`gadget_routing.py`):
Read block offsets from `qubit_allocation.block_offsets` and use them
to determine grid layout direction. For vertical stacking, place blocks
in vertically stacked rows instead of side-by-side columns:

```python
def allocate_block_regions(
    block_names, qubits_per_block, k, gap_cols=0,
    layout_direction="column",  # "column" (vertical) or "row" (horizontal)
):
    if layout_direction == "column":
        # Stack blocks vertically (increasing rows)
        row_cursor = 0
        for name in block_names:
            n_qubits = qubits_per_block[name]
            n_cols = int(math.ceil(math.sqrt(n_qubits / k)))
            n_rows = int(math.ceil(n_qubits / (n_cols * k)))
            regions[name] = (row_cursor, 0, row_cursor + n_rows, n_cols)
            row_cursor += n_rows + gap_rows
    else:
        # Original side-by-side (horizontal)
        ...
```

**Fix I2 — Bridge ancilla placement at block boundaries**
(`gadget_routing.py`/`architectures.py`):
When building per-block ion lists, assign bridge ancilla ions to
boundary-adjacent traps in the relevant block's sub-grid. Currently
`regularPartition` treats all ions equally — bridge ions could end up
deep inside the block rather than at the boundary.

Add bridge ion hints to `regularPartition`:

```python
# Before regularPartition:
boundary_ions = {ion: "edge_bottom" for ion in bridge_ions_for_this_block}
# In regularPartition: prefer placing boundary_ions in edge-adjacent traps
```

**Files**: `gadget_routing.py` (`allocate_block_regions`), `architectures.py` (`build_topology_per_block`)

---

### Issue J — Poor Batching of Resets/Rotations

**Root cause**: `_drain_single_qubit_ops()` groups by `_TYPE_ORDER`:
```python
_TYPE_ORDER = {XRotation: 0, YRotation: 1, QubitReset: 2, Measurement: 3}
```

This means:
1. All XRotation gates drain first (even if they should come AFTER reset)
2. Resets drain third (should be first for preparation sequences)
3. Within a type, cross-epoch ops on disjoint qubits ARE merged (good)

**Fix J1 — Physical-order type priority**:
```python
_TYPE_ORDER = {QubitReset: 0, XRotation: 1, YRotation: 2, Measurement: 3}
```

**Fix J2 — Commuting-type parallel batching**:
RX on qubit A and RY on qubit B can execute in parallel since they're on
different qubits. Instead of strict type-priority drain, group by
**disjoint-qubit compatibility**:

```python
# Instead of draining all XRotation then all YRotation:
# Drain any rotation type as long as ions are disjoint
rotation_types = {XRotation, YRotation}
if best_type in rotation_types:
    # Also drain other rotation types on disjoint ions
    for q in ion_queues.values():
        if q and type(q[0]) in rotation_types:
            # Include in this batch
```

**Files**: `qccd_WISE_ion_route.py` (`_TYPE_ORDER`, `_drain_single_qubit_ops`)

---

### Issue K — First CX Round Not Nearest-Neighbour

**Root cause**: `hillClimbOnArrangeClusters()` clusters ions by general
locality, not by the specific EC CX schedule's required adjacency pattern.
The first EC round's CX pairs may involve ions in non-adjacent traps.

**Fix K1 — CX-schedule-aware clustering** (`architectures.py`):
When building per-block topology, pass the first EC round's CX pairs as
clustering hints. Ions that form CX pairs should be clustered in adjacent
traps:

```python
def build_topology_per_block(self, ..., cx_adjacency_hints=None):
    # After regularPartition, before hillClimb:
    # Seed the hill-climb with CX pair adjacency as the objective
    if cx_adjacency_hints:
        for (ion_a, ion_b) in cx_adjacency_hints:
            # Penalize non-adjacent trap placement
```

This is a significant optimization but not blocking — the SAT solver will
fix non-adjacent pairs via reconfiguration.

**Files**: `architectures.py` (`build_topology_per_block`, `hillClimb`)

---

### Issue L — Missing Pre-Gadget EC Rounds

**Root cause**: For CSS surgery (flat codes), `get_blocks_to_skip_pre_rounds()`
returns ALL blocks. This is **by design** — the merge phases handle EC
internally. The stim circuit confirms this: no standalone EC rounds
before the first ZZ merge CX.

The interleaved EC fix (already applied) adds EC rounds within each merge
phase. So the "missing d EC rounds" are actually present — they're just
interleaved with the bridge CX rather than appearing as a separate phase.

**Status**: Already fixed by interleaved EC implementation. No further
change needed.

---

### Issue M — Global Reconfigs Not Merged

**Root cause**: `_compute_transition_reconfig_steps()` generates per-block
transition steps. These are returned as separate RoutingSteps. The
execution loop applies them sequentially.

For **disjoint blocks**, transition reconfigs can run in parallel:
block_0's reconfig doesn't affect block_1's region. The existing
`_merge_disjoint_block_schedules()` handles this for **intra-round**
merging but is not applied to **inter-phase transitions**.

**Fix M1 — Apply block-level merging to transition steps**
(`gadget_routing.py`):

```python
# In _compute_transition_reconfig_steps():
per_block_steps = [solve_transition(block) for block in blocks]
merged_step = _merge_block_routing_steps(per_block_steps)
return [merged_step]  # Single step instead of N
```

**Fix M2 — Coalesce in execution loop** (same as Fix H1).

**Files**: `gadget_routing.py`, `qccd_WISE_ion_route.py`

---

## Implementation Priority & Dependency Graph

```
                    ┌─────────────────────────────┐
                    │  FIX E (CRITICAL)            │
                    │  Multi-qubit instruction     │
                    │  processing in decompose     │
                    │  (fixes E, G, partial D)     │
                    └──────────┬──────────────────┘
                               │
                    ┌──────────▼──────────────────┐
                    │  FIX E1 (_TYPE_ORDER)        │
                    │  Reset before rotations      │
                    │  (fixes E, J partial)        │
                    └──────────┬──────────────────┘
                               │
              ┌────────────────┤
              │                │
    ┌─────────▼────────┐  ┌───▼──────────────────┐
    │  FIX A (layout)  │  │  FIX I (grid alloc)  │
    │  Adaptive viewBox │  │  Vertical stacking   │
    │  (fixes A, B, C) │  │  + bridge placement   │
    └──────────────────┘  └───┬──────────────────┘
                              │
                    ┌─────────▼──────────────────┐
                    │  FIX D (highlighting)       │
                    │  Stim-order MS execution    │
                    └────────────────────────────┘
                               │
              ┌────────────────┼────────────────┐
              │                │                │
    ┌─────────▼──────┐  ┌─────▼─────┐  ┌───────▼─────┐
    │  FIX H/M       │  │  FIX J    │  │  FIX K      │
    │  Reconfig merge │  │  Batching │  │  CX-aware   │
    └────────────────┘  └───────────┘  │  clustering  │
                                       └─────────────┘
```

### Ordered Task List

| Step | Fix | Effort | Impact | Files |
|------|-----|--------|--------|-------|
| **1** | **E3** — Multi-qubit instruction loop in `decompose_to_native()` | Medium | CRITICAL — fixes dropped gates/measurements | `trapped_ion_compiler.py` |
| **2** | **E2** — Direct `RX`/`MX`/`MRX` handling (avoid fallback) | Small | HIGH — ensures CSS-specific gates compile correctly | `trapped_ion_compiler.py` |
| **3** | **E1/J1** — Fix `_TYPE_ORDER` (Reset before Rotation) | Small | HIGH — correct physical execution order | `qccd_WISE_ion_route.py` |
| **4** | **A1/A2** — Adaptive viewBox + square padding | Medium | HIGH — fixes timeslice visual issues | `viz/stim_panel.py`, `viz/animation.py` |
| **5** | **I1** — Vertical grid allocation | Medium | HIGH — matches stim layout direction | `gadget_routing.py` |
| **6** | **I2** — Bridge ancilla boundary placement | Hard | MEDIUM — improves merge routing locality | `architectures.py` |
| **7** | **D1** — Stim-order MS execution preference | Medium | MEDIUM — correct highlighting | `qccd_WISE_ion_route.py` |
| **8** | **H1/M1** — Coalesce adjacent reconfig steps | Medium | MEDIUM — fewer animation frames | `qccd_WISE_ion_route.py`, `gadget_routing.py` |
| **9** | **J2** — Commuting rotation parallel batching | Medium | LOW — slight parallelism improvement | `qccd_WISE_ion_route.py` |
| **10** | **K1** — CX-schedule-aware clustering | Hard | LOW — optimization only | `architectures.py` |

---

## Detailed Fix Locations

### Step 1 — Multi-Qubit Instruction Loop (CRITICAL)

**File**: `src/qectostim/experiments/hardware_simulation/trapped_ion/utils/trapped_ion_compiler.py`

**Location**: `decompose_to_native()`, lines ~410-530.

**Current code** (problematic):
```python
# Line ~416-420: Only extracts first qubit
try:
    idx = int(qubit_parts[0])
    ion = self._ion_mapping[idx][0]
except (ValueError, KeyError):
    continue

if gate_name in ("M", "MZ"):
    operations.append(_tag(Measurement.qubitOperation(ion), _origin, _epoch))
elif gate_name == "H":
    ...
elif gate_name == "R":
    operations.append(_tag(QubitReset.qubitOperation(ion), _origin, _epoch))
```

**Fix**: For all single-qubit gate types (`M`, `MZ`, `R`, `H`, `MR`),
iterate over ALL `qubit_parts`:

```python
if gate_name in ("M", "MZ"):
    for qi in qubit_parts:
        idx = int(qi)
        ion = self._ion_mapping[idx][0]
        operations.append(_tag(Measurement.qubitOperation(ion), _origin, _epoch))

elif gate_name == "H":
    for qi in qubit_parts:
        idx = int(qi)
        ion = self._ion_mapping[idx][0]
        operations.extend([
            _tag(YRotation.qubitOperation(ion), _origin, _epoch),
            _tag(XRotation.qubitOperation(ion), _origin, _epoch),
        ])

elif gate_name == "R":
    for qi in qubit_parts:
        idx = int(qi)
        ion = self._ion_mapping[idx][0]
        operations.append(_tag(QubitReset.qubitOperation(ion), _origin, _epoch))
    if self.data_qubit_idxs is None:
        dataQubits.clear()

elif gate_name == "MR":
    for qi in qubit_parts:
        idx = int(qi)
        ion = self._ion_mapping[idx][0]
        operations.append(_tag(Measurement.qubitOperation(ion), _origin, _epoch))
        operations.append(_tag(QubitReset.qubitOperation(ion), _origin, _epoch))
```

For **2-qubit gates** (`CNOT`, `CX`, `CZ`), iterate over pairs:

```python
elif gate_name in ("CNOT", "CX", "ZCX"):
    for pair_start in range(0, len(qubit_parts), 2):
        idx = int(qubit_parts[pair_start])
        idx2 = int(qubit_parts[pair_start + 1])
        ion = self._ion_mapping[idx][0]
        ion2 = self._ion_mapping[idx2][0]
        operations.extend([...])
```

### Step 2 — Direct RX/MX/MRX Handling

Add explicit handlers before the fallback block (line ~465):

```python
elif gate_name == "RX":
    for qi in qubit_parts:
        idx = int(qi)
        ion = self._ion_mapping[idx][0]
        operations.append(_tag(QubitReset.qubitOperation(ion), _origin, _epoch))
        operations.append(_tag(YRotation.qubitOperation(ion), _origin, _epoch))
        operations.append(_tag(XRotation.qubitOperation(ion), _origin, _epoch))

elif gate_name in ("MX", "MY"):
    for qi in qubit_parts:
        idx = int(qi)
        ion = self._ion_mapping[idx][0]
        # Basis change then measure
        if gate_name == "MX":
            operations.append(_tag(XRotation.qubitOperation(ion), _origin, _epoch))
            operations.append(_tag(YRotation.qubitOperation(ion), _origin, _epoch))
        else:  # MY
            operations.append(_tag(YRotation.qubitOperation(ion), _origin, _epoch))
        operations.append(_tag(Measurement.qubitOperation(ion), _origin, _epoch))

elif gate_name == "MRX":
    for qi in qubit_parts:
        idx = int(qi)
        ion = self._ion_mapping[idx][0]
        # MX part
        operations.append(_tag(XRotation.qubitOperation(ion), _origin, _epoch))
        operations.append(_tag(YRotation.qubitOperation(ion), _origin, _epoch))
        operations.append(_tag(Measurement.qubitOperation(ion), _origin, _epoch))
        # RX part
        operations.append(_tag(QubitReset.qubitOperation(ion), _origin, _epoch))
        operations.append(_tag(YRotation.qubitOperation(ion), _origin, _epoch))
        operations.append(_tag(XRotation.qubitOperation(ion), _origin, _epoch))
```

### Step 3 — Fix `_TYPE_ORDER`

**File**: `src/qectostim/experiments/hardware_simulation/trapped_ion/compiler/qccd_WISE_ion_route.py`

**Current** (~line 1985):
```python
_TYPE_ORDER = {XRotation: 0, YRotation: 1, QubitReset: 2, Measurement: 3}
```

**Fix**:
```python
_TYPE_ORDER = {QubitReset: 0, XRotation: 1, YRotation: 2, Measurement: 3}
```

---

## Testing Strategy

### Unit tests (no SAT):

```bash
# Test multi-qubit instruction processing
PYTHONPATH=src my_venv/bin/python -c "
from qectostim.gadgets.css_surgery_cnot import CSSSurgeryCNOTGadget
from qectostim.codes.surface.rotated_surface import RotatedSurfaceCode
from qectostim.experiments.ft_gadget_experiment import FaultTolerantGadgetExperiment
from qectostim.experiments.hardware_simulation.trapped_ion.utils import *

g = CSSSurgeryCNOTGadget()
code = RotatedSurfaceCode(distance=2)
ft = FaultTolerantGadgetExperiment(codes=[code], gadget=g, noise_model=None,
                                    num_rounds_before=2, num_rounds_after=2)
circ = ft.to_stim()

# Count  gates in stim circuit
import re
stim_lines = str(circ).split('\n')
n_R = sum(len(re.findall(r'\d+', line)) for line in stim_lines
          if line.strip().startswith('R '))
n_RX = sum(len(re.findall(r'\d+', line)) for line in stim_lines
           if line.strip().startswith('RX '))
n_M = sum(len(re.findall(r'\d+', line)) for line in stim_lines
          if line.strip().startswith('M '))
n_MX = sum(len(re.findall(r'\d+', line)) for line in stim_lines
           if line.strip().startswith('MX '))

print(f'Stim R count: {n_R}')
print(f'Stim RX count: {n_RX}')
print(f'Stim M count: {n_M}')
print(f'Stim MX count: {n_MX}')
# After fix, compiled native ops should have matching counts
"
```

### Integration test (with SAT):

```bash
WISE_INPROCESS_LIMIT=999999999 PYTHONPATH=src my_venv/bin/python \
    _test_regression_tcnot.py
```

### Full animation test:

Run notebook cell 24 and verify:
1. ✅ All qubits get reset (not just first)
2. ✅ All qubits get measured at end
3. ✅ RX appears as Reset+YRot+XRot (3 ops per qubit)
4. ✅ Timeslice viewBox shows readable block layout
5. ✅ First MS round highlights first CX layer
6. ✅ No phantom reconfigs between phases

---

## Risk Assessment

| Fix | Risk | Mitigation |
|-----|------|------------|
| E3 (multi-qubit loop) | May change operation count, breaking `toMoveIdx` tracking | Test: verify `toMoveOps` are still built correctly for CX |
| E1 (_TYPE_ORDER) | Changes execution order of ALL circuits, not just CSS | Test: regression test TransversalCNOT |
| A1 (adaptive viewBox) | May cause SVG jitter between ticks | Use smoothed viewBox (exponential moving average) |
| I1 (vertical grid) | May break d>2 cases where horizontal layout was needed | Keep `layout_direction` parameter with default matching stim layout |
| D1 (stim-order MS) | Routing constraints may prevent stim-order execution | Keep scan-all-remaining as fallback when stim-order pair not found |
