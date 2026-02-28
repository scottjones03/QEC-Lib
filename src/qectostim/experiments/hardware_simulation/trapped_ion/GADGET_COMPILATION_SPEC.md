# WISE Gadget Compilation — Implementation Specification

## 1. Overview

Extend the WISE trapped-ion compilation pipeline to support **fault-tolerant gadget experiments** (e.g. TransversalCNOT) with intelligent temporal and spatial slicing that makes large-distance gadgets feasible.

### Core Insight

A gadget experiment has a predictable temporal structure — an alternating sequence of EC rounds and gadget phases:

```
              TransversalCNOT (2 blocks, 1 gadget phase)
[Init] → [EC × d] → [Gadget] → [EC × d] → [Measure]
         ^^^^^^^^    ^^^^^^^^    ^^^^^^^^
         2 blocks    blk0↔blk1   2 blocks
         independent merge grid  independent

              KnillEC (3 blocks, 3 gadget phases)
[Init] → [EC × d] → [Phase2: bell_a↔bell_b] → [EC × d] → [Phase3: data↔bell_a] → [EC × d] → [Measure]
         ^^^^^^^^    ^^^^^^^^^^^^^^^^^^^^^^     ^^^^^^^^    ^^^^^^^^^^^^^^^^^^^^^^   ^^^^^^^^
         3 blocks    merge 2, EXCLUDE data_blk  3 blocks    merge 2, EXCLUDE bell_b 3 blocks
         each d²-1   SAT grid = 2(2d²-1) ions   reuse cache SAT grid = 2(2d²-1) ions reuse cache

              CSSSurgeryCNOT (3 blocks, 5 gadget phases)
[Init] → [EC × d] → [ZZ_merge: blk0↔blk1] → [EC] → [ZZ_split] → [EC] → [XX_merge: blk1↔blk2] → [EC] → [XX_split] → [EC] → [ANC_MX] → [Measure]
```

**Three optimisation axes**:

1. **Temporal slicing**: EC rounds are repeated `d` times with identical structure per block — route once, replay `d-1` times.
2. **Spatial slicing — Level 1 (block-level)**: Choose which blocks are on the grid at all.
   - During EC: each block routes on its *own* sub-grid (other blocks absent).
   - During a gadget phase: only the *interacting* blocks are on the grid. Idle blocks are **completely excluded** — their ions consume zero SAT variables, zero grid cells, zero solver time.
3. **Spatial slicing — Level 2 (patch-and-route within a grid)**: Even after Level-1 slicing, a single block's grid can still be too large for one SAT call (e.g. d=12 → 287 ions on a ~17×17 grid). The existing `_patch_and_route()` algorithm further subdivides any grid into overlapping constant-size sub-patches (configurable `subgrid_width × subgrid_height`), solves each patch locally via `optimal_QMR_for_WISE`, and stitches results via checkerboard tiling with boundary-preference heuristics.

**The two levels compose hierarchically**: Level 1 selects *which ions* are on the grid; Level 2 partitions *that* grid into tractable SAT sub-problems. Both levels are essential:

```
           Level 1                         Level 2
    (block-level slicing)           (patch-and-route slicing)
    ┌─────────────────┐             ┌────┬────┬────┐
    │ Select blocks   │             │ p1 │ p2 │ p3 │   patch_w × patch_h
    │ for this phase  │  ──────►    ├────┼────┼────┤   sub-patches tiled
    │ (exclude idle)  │             │ p4 │ p5 │ p6 │   across the grid
    └─────────────────┘             └────┴────┴────┘
    Grid: N_interacting × d²-1       Each patch: ≤ subgrid_width × subgrid_height
    ions                             ions → single SAT call
```

This is especially critical for cross-block gadget phases. For example, a d=12 transversal CNOT merges 2 blocks onto a grid of ~574 ions (~24×24 traps). Without Level-2 patch slicing, a single SAT call on 574 ions would be intractable. With `subgrid_width=4, subgrid_height=3`, the grid is tiled into ~48 small patches, each solved independently.

Level 2 is **already implemented** (`_patch_and_route()` in `qccd_WISE_ion_route.py`). The gadget compilation work only adds Level 1 on top.

### Supported Gadgets

| Gadget | Blocks | Gadget Phases | Interaction Pattern |
|--------|--------|---------------|---------------------|
| `TransversalCNOTGadget` | 2 (block_0, block_1) | 1 | blk0 ↔ blk1 |
| `CZHTeleportGadget` | 2 (data, ancilla) | 3 | data ↔ anc (prep, gate, meas) |
| `CNOTHTeleportGadget` | 2 (data, ancilla) | 3 | data ↔ anc (prep, gate, meas) |
| `KnillECGadget` | 3 (data, bell_a, bell_b) | 3 | Ph2: bell_a↔bell_b, Ph3: data↔bell_a |
| `CSSSurgeryCNOTGadget` | 3 (blk0, blk1, blk2) | 5 | Ph1: blk0↔blk1, Ph3: blk1↔blk2 |

**Key property**: No gadget has >2 blocks interacting simultaneously in a single phase. All multi-block interactions are pairwise — but *different pairs* interact in *different phases*.

### Scaling Target

**TransversalCNOT (2 blocks)**:

| Distance | Qubits | MS/EC round | Gadget MS | Total MS | Naïve SAT calls | Optimised SAT calls |
|----------|--------|-------------|-----------|----------|-----------------|---------------------|
| d=3      | 34     | 48          | 9         | 297      | 25              | ~6                  |
| d=5      | 98     | 160         | 25        | 1,625    | 41              | ~6                  |
| d=7      | 194    | 336         | 49        | 4,753    | 57              | ~6                  |
| d=12     | 574    | 1,056       | 144       | 25,488   | 97              | ~6                  |

Optimised = 1 EC round per block (2) + 1 gadget round (1) + transitions (2) + restore (1) = **~6**.

**KnillEC (3 blocks, 3 gadget phases)** — spatial slicing impact:

| Distance | Qubits | Naïve SAT calls | Optimised SAT calls | SAT grid per gadget phase |
|----------|--------|-----------------|---------------------|---------------------------|
| d=3      | 51     | ~39             | ~10                 | 34 ions (2 of 3 blocks)   |
| d=5      | 147    | ~63             | ~10                 | 98 ions (2 of 3 blocks)   |

Optimised = 1 EC round per block (3) + 2 gadget phases (2) + transitions (4) + restore (1) = **~10**.
Critically, each gadget phase SAT solve uses only **2/3 of the ions** — the idle block is excluded entirely.

**CSSSurgeryCNOT (3 blocks, 5 gadget phases)**:

| Distance | Qubits | Naïve SAT calls | Optimised SAT calls | SAT grid per gadget phase |
|----------|--------|-----------------|---------------------|---------------------------|
| d=3      | 51     | ~55             | ~16                 | 34 ions (2 of 3 blocks)   |
| d=5      | 147    | ~95             | ~16                 | 98 ions (2 of 3 blocks)   |

Optimised = 1 EC round per block (3) + 5 gadget phases (5) + transitions (6) + restore (2) = **~16**.

### Complexity Reduction via Two-Level Hierarchical Spatial Slicing

**Level 1 — Block-level slicing** (which ions are on the grid):

```
       Naïve (all ions on one grid)          Level 1: per-phase block slicing
       ┌──────────────────────────┐          ┌─────────────┐  ┌─────────────┐
       │  blk0   blk1   blk2     │          │ blk0  blk1  │  │   blk2      │
       │  17     17     17       │          │ 17    17    │  │   EXCLUDED  │
       │  = 51 ions, huge grid   │          │ = 34 ions   │  │   (idle)    │
       └──────────────────────────┘          └─────────────┘  └─────────────┘
       Grid: 51 ions, 3× block width        Grid: 34 ions, 2× block width
```

**Level 2 — Patch-and-route slicing** (within the selected grid):

```
       Without L2 (single SAT call)          With L2: _patch_and_route tiling
       ┌─────────────────────────┐           ┌────┬────┬────┐
       │                         │           │ p1 │ p2 │ p3 │  subgrid 4×3
       │  34 ions, 1 huge SAT    │           ├────┼────┼────┤  patches
       │  intractable at d≥7     │           │ p4 │ p5 │ p6 │  each ≤12 ions
       │                         │           ├────┼────┼────┤
       │                         │           │ p7 │ p8 │ p9 │
       └─────────────────────────┘           └────┴────┴────┘
       O(34³) = intractable                  9 × O(12³) = fast
```

**Combined effect at scale** (d=12 KnillEC, 3 blocks):

| Approach | Grid ions | Grid traps | SAT calls | Ions per SAT call | Feasible? |
|----------|-----------|------------|-----------|-------------------|-----------|
| Naïve (all ions, no slicing) | 861 | ~30×30 | 1 giant | 861 | ❌ Intractable |
| Level 1 only (block slicing) | 574 | ~24×24 | 1 large | 574 | ❌ Still intractable |
| Level 2 only (patch slicing, all ions) | 861 | ~30×30 | ~75 patches | ~12 each | ⚠️ Works but huge grid |
| **Both levels** | 574 | ~24×24 | ~48 patches | ~12 each | ✅ Fast, small grid |
| **Both levels + EC cache** | 287 (1 block) | ~17×12 | ~14 patches | ~12 each | ✅ Fastest |

Level 1 reduces the grid (fewer ions, fewer columns); Level 2 makes any grid tractable. Together they minimise both the number of patches AND the grid the patches tile over.

---

## 2. Architecture

### 2.1 Phase-Aware Routing Pipeline

```
                              QECMetadata
                                  │
                    ┌─────────────┴──────────────┐
                    │                            │
              PhaseDecomposer              BlockPartitioner
                    │                            │
          ┌─────────┴──────────────┐    ┌────────┴────────────────┐
          │                        │    │                         │
    for each phase in              │    SubGrid:block_0           │
    QECMetadata.phases:            │    SubGrid:block_1           │
          │                        │    SubGrid:block_2  (if N≥3) │
    ┌─────┴──────┐                 │    │                         │
    │            │                 │    │                         │
  EC phase    Gadget phase         │    │                         │
    │            │                 │    │                         │
    ▼            ▼                 │    │                         │
  Per-block    Merge ONLY          │    │                         │
  sub-grids    interacting         │    │                         │
  (parallel)   blocks;             │    │                         │
    │          EXCLUDE idle        │    │                         │
    │          blocks from grid    │    │                         │
    ▼            ▼                 │    │                         │
  CachedRoute  PartialMergeRoute   │    │                         │
    │            │                 │    │                         │
    └──────┬─────┘                 │    │                         │
           │                       │    │                         │
     TransitionRouter (per boundary)    │                         │
           │                       │    │                         │
    ┌──────┴─────────────────────────────┘                        │
    │                                                             │
    UnifiedSchedule ◄─────────────────────────────────────────────┘
    │
    StimCircuit + Noise + Decode
```

### 2.2 Key New Abstractions

#### `PhaseRoutingPlan`

```python
@dataclass
class PhaseRoutingPlan:
    """Plan for routing one temporal phase of a gadget experiment."""
    phase_type: str                    # "ec", "gadget", "transition"
    phase_index: int                   # index into QECMetadata.phases
    interacting_blocks: List[str]      # blocks that interact (merged onto shared grid)
    idle_blocks: List[str]             # blocks EXCLUDED from this phase's SAT grid
    all_blocks: List[str]              # interacting + idle (for bookkeeping)
    ms_pairs_per_round: List[List[Tuple[int, int]]]  # ion-index pairs per MS round
    num_rounds: int                    # how many times this repeats
    is_cached: bool                    # True if solved once and replayed
    grid_region: Tuple[int, int, int, int]  # (r0, c0, r1, c1) sub-grid for this phase
    round_signature: Optional[Tuple]   # canonical key for cache lookup
    identical_to_phase: Optional[int]  # phase index with same routing solution
```

**Idle block exclusion**: When `idle_blocks` is non-empty, those blocks' ions are not placed on the SAT grid. Their ions remain parked in their last known positions (from the preceding EC round). No SAT variables are allocated for them, no grid cells are consumed, and no reconfigurations are planned for them. This is the primary mechanism for reducing SAT complexity in multi-block gadgets.

#### `BlockSubGrid`

```python
@dataclass
class BlockSubGrid:
    """Sub-grid allocation for one logical block."""
    block_name: str
    grid_region: Tuple[int, int, int, int]  # (r0, c0, r1, c1) on the full grid
    n_rows: int                              # sub-grid row count
    n_cols: int                              # sub-grid column count
    ion_indices: List[int]                   # ions assigned to this sub-grid
    qubit_to_ion: Dict[int, int]             # stim qubit idx → ion idx
    initial_layout: np.ndarray               # n_rows × n_cols ion arrangement
```

### 2.3 Routing Engine Extraction (`_route_round_sequence`)

The working routing engine in `ionRoutingWISEArch()` bundles two concerns:

1. **Routing logic** (pure arrays): windowed SAT solving, block caching, BT propagation, patch decomposition, reconfig merging, prev_pmax warm-start, graceful error recovery
2. **Execution logic** (arch-coupled): `_apply_layout_as_reconfiguration()`, 1-qubit operation scheduling, MS gate execution, barrier tracking

We extract concern (1) into a new function **`_route_round_sequence()`** in `qccd_WISE_ion_route.py` so both `ionRoutingWISEArch` (single-block) and `gadget_routing.py` (multi-block orchestrator) can call it. All 10 tricks from the working code are preserved inside this extraction.

#### Interface

```python
@dataclass
class RoutingStep:
    """One reconfiguration + MS-gate step produced by the routing engine."""
    layout_after: np.ndarray           # target ion arrangement after reconfig
    schedule: Optional[List[Dict]]     # swap passes to achieve layout_after (None if from cache)
    solved_pairs: List[Tuple[int,int]] # MS pairs enabled by this layout
    ms_round_index: int                # which parallelPairs round these pairs solve
    from_cache: bool                   # True if replayed from block cache
    tiling_meta: Tuple[int,int]        # (cycle_idx, tiling_idx) for debugging
    can_merge_with_next: bool          # hint for reconfig merge optimisation

def _route_round_sequence(
    oldArrangementArr: np.ndarray,
    wiseArch: QCCDWiseArch,
    parallelPairs: List[List[Tuple[int,int]]],
    *,
    lookahead: int,
    subgridsize: Tuple[int,int,int],
    base_pmax_in: int,
    active_ions: Set[int],
    BTs: Optional[List[Dict[int, Tuple[int,int]]]] = None,
    toMoveOps: Optional[List] = None,          # for block-cache key building
    stop_event: Optional[threading.Event] = None,
    progress_callback: Optional[Callable] = None,
    max_inner_workers: Optional[int] = None,
) -> Tuple[List[RoutingStep], np.ndarray]:
    """Pure-array routing engine with all 10 working-code tricks.

    Extracts the main-loop logic from ionRoutingWISEArch without any
    arch or Operation coupling. Preserves all variable names.

    Returns:
        steps: ordered list of RoutingSteps to be applied
        final_layout: the arrangement array after all steps
    """
```

#### What Lives Inside `_route_round_sequence` (all 10 tricks)

| # | Trick | Implementation detail (variable names preserved) |
|---|-------|--------------------------------------------------|
| 1 | Gating-capacity overflow spilling | `_compute_patch_gating_capacity()` → `max_pairs`; excess pairs deferred to next tiling step |
| 2 | Cross-boundary preference soft clauses | `_compute_cross_boundary_prefs()` → `cross_boundary_prefs` passed to `_optimal_QMR_for_WISE` |
| 3 | BT pin conflict detection | `start_row`/`target_col` uniqueness check; conflicting BTs dropped with warning |
| 4 | Multi-round lookahead BT pins | Forward-visibility: BTs from rounds `[idx+1 .. idx+lookahead]` passed as soft pins |
| 5 | Reconfig merge optimisation | `_ions_unmoved()` + `_merge_reconfig_schedules()` applied to consecutive RoutingSteps; sets `can_merge_with_next` |
| 6 | Block caching with per-offset replay | `block_cache` dict keyed by `tuple(sorted(pp) for pp in P_arr)`, BLOCK_LEN=4, `recheck_cache` flag, `blk_idx`/`blk_end` cursors |
| 7 | `_rebuild_schedule_for_layout` | Called when target layout known but schedule missing (cache replay); iterative convergent solver |
| 8 | `prev_pmax` warm-starting | Tracks best `pmax_in` across rounds; seeds next `_patch_and_route` call |
| 9 | `NoFeasibleLayoutError` graceful recovery | Catches exception → spills excess pairs → retries with fewer pairs per patch |
| 10 | Single-qubit scheduling metadata | Attaches epoch/rotation metadata to RoutingSteps so the execution layer can schedule 1q ops between reconfigs |

#### Execution Layer (stays in `ionRoutingWISEArch` / new `ionRoutingGadgetArch`)

The execution layer iterates over the `List[RoutingStep]` returned by `_route_round_sequence()` and for each step:

```python
for step in routing_steps:
    # 1. Apply reconfiguration (arch-coupled bridge)
    _apply_layout_as_reconfiguration(arch, wiseArch, oldArr, step.layout_after,
                                     allOps, step.schedule)
    oldArr = step.layout_after

    # 2. Schedule single-qubit ops (epoch-aware, existing code)
    schedule_1q_ops_for_epoch(step.ms_round_index, allOps, allBarriers, ...)

    # 3. Execute MS gates (existing code)
    execute_ms_gates(step.solved_pairs, allOps, allBarriers, ...)
```

This separation means:
- **`ionRoutingWISEArch`** is refactored to call `_route_round_sequence()` → iterate → apply. Same API, same output, zero behavioral change.
- **`gadget_routing.py`** calls `_route_round_sequence()` per block (EC phases) or per merged grid (gadget phases), collecting `List[RoutingStep]` per phase.
- A new **`ionRoutingGadgetArch()`** entry point takes the combined plan from `gadget_routing.py` and executes it using the global `QCCDArchitecture` to produce `(allOps, barriers, reconfigTime)`.

#### Data Flow Diagram

```
                    ionRoutingWISEArch (single-block, unchanged API)
                    ┌──────────────────────────────────────────────┐
                    │  build parallelPairs from operations          │
                    │  build initial layout                         │
                    │  ↓                                           │
                    │  _route_round_sequence(...)                   │
                    │  ↓                                           │
                    │  List[RoutingStep]                            │
                    │  ↓                                           │
                    │  for step in steps:                           │
                    │    _apply_layout_as_reconfiguration(...)      │
                    │    schedule_1q_ops(...)                       │
                    │    execute_ms(...)                            │
                    │  ↓                                           │
                    │  return (allOps, barriers, reconfigTime)      │
                    └──────────────────────────────────────────────┘

                    ionRoutingGadgetArch (multi-block, new entry point)
                    ┌──────────────────────────────────────────────┐
                    │  build per-block sub-grids                    │
                    │  ↓                                           │
                    │  gadget_routing.route_full_experiment(...)    │
                    │    ├─ EC phase:  per-block _route_round_seq  │
                    │    ├─ Gadget:    merged-grid _route_round_seq│
                    │    └─ Transition: BT-embedded in gadget      │
                    │  ↓                                           │
                    │  Dict[phase_id, List[RoutingStep]]            │
                    │  ↓                                           │
                    │  for phase in phases:                         │
                    │    for step in phase_steps:                   │
                    │      _apply_layout_as_reconfiguration(...)    │
                    │      schedule_1q_ops(...)                     │
                    │      execute_ms(...)                          │
                    │  ↓                                           │
                    │  return (allOps, barriers, reconfigTime)      │
                    └──────────────────────────────────────────────┘
```

---

## 3. Detailed Design

### 3.1 Grid Composition from Block Layouts

**Source of truth**: `GadgetLayout.compute_layout()` / `QubitAllocation` from the gadget.

The existing gadget layout system places blocks along the x-axis with a margin. We use the gadget's own `QubitAllocation` to determine:
1. Which qubit indices belong to which block
2. The spatial offset of each block (from `block.offset`)
3. Data vs ancilla classification (from `block.data_range`, `block.x_anc_range`, `block.z_anc_range`)

**Grid sizing per block**: Each block with `n_qubits = 2d² - 1` ions gets a sub-grid:
```python
n_rows_block = int(math.ceil(math.sqrt(n_qubits)))
n_cols_block = int(math.ceil(n_rows_block / k))  # k = trap_capacity
```

**Per-phase grid composition (Level 1 slicing)**: The grid is NOT built once for all ions. Instead, the grid is built per-phase with only the participating blocks:

```python
def build_phase_grid(interacting_blocks: List[BlockSubGrid], k: int) -> QCCDWiseArch:
    """Build a grid containing ONLY the interacting blocks for a phase.
    
    This is Level 1 spatial slicing: select which blocks (and therefore
    which ions) are on the grid at all. Level 2 (_patch_and_route) then
    tiles THIS grid into sub-patches.
    """
    # Arrange blocks side-by-side along x-axis
    n_rows = max(b.n_rows for b in interacting_blocks)
    n_cols = sum(b.n_cols for b in interacting_blocks) + len(interacting_blocks) - 1  # gap cols
    m_traps = n_cols * k
    return QCCDWiseArch(m=m_traps, n=n_rows, k=k)
```

**Grid types** used in different contexts:

| Context | Level 1 (blocks on grid) | Grid size | Level 2 (patch tiling) |
|---------|-------------------------|-----------|------------------------|
| EC round (per-block) | 1 block only | Block sub-grid (~d²-1 ions) | `_patch_and_route` tiles into `subgrid_w × subgrid_h` patches |
| Gadget phase (partial merge) | 2 interacting blocks | 2-block merged grid (~2(d²-1) ions) | `_patch_and_route` tiles the merged grid |
| Transition | Interacting blocks | Same as gadget grid | BT-embedded or dedicated reconfig |

**For N-block gadgets** (e.g. KnillEC with 3 blocks), a gadget phase that involves bell_a ↔ bell_b builds a grid with **only** those 2 blocks. `data_block`'s ions are not placed on the grid, not in the SAT variable space, and not considered for reconfiguration. Then `_patch_and_route` further tiles that 2-block grid into patches.

**Important**: `data_qubit_idxs` MUST be passed to `TrappedIonCompiler` for multi-block circuits — the QUBIT_COORDS parity heuristic breaks with x-offsets. Extract data qubit indices from `QubitAllocation`:
```python
data_qubit_idxs = set()
for block in qubit_allocation.blocks.values():
    data_qubit_idxs.update(block.data_range)
```

### 3.1.1 Disjoint Block Layout Invariant

**Critical requirement**: Level 1 spatial slicing only works if different blocks occupy **completely disjoint** regions of the WISE trap grid. When blocks share traps, it is impossible to slice out an independent sub-grid per block — the SAT solver would need all ions on one shared grid, defeating the purpose of Level 1 slicing.

#### The Problem with Naïve Mapping

The existing `map_qubits()` pipeline calls `regularPartition()` with **all** measurement and data ions from **all** blocks simultaneously, then `hillClimbOnArrangeClusters()` maps the resulting clusters to grid positions across the entire `m × n` architecture. This produces a globally optimal initial layout, but ions from different blocks become **interleaved** across the grid — a block_0 ion might share a trap with a block_1 ion. Under this mapping, Level 1 spatial slicing is impossible.

```
    Naïve mapping (all ions together):
    ┌───────────────────────────────────┐
    │ blk0  blk1  blk0  blk1  blk0     │  ← ions interleaved across traps
    │  D0    D5    M1    D6    M2       │  ← cannot extract disjoint sub-grids
    │ blk1  blk0  blk1  blk0  blk1     │
    │  D7    M3    D8    D4    M9       │
    └───────────────────────────────────┘
```

#### Per-Block Qubit-to-Ion Mapping

Instead, we run the cluster partitioning + Hungarian assignment pipeline **per block**, each block mapping onto its own **allocated sub-region** of the full grid:

```python
def map_qubits_per_block(
    block_sub_grids: Dict[str, BlockSubGrid],
    measurement_ions_per_block: Dict[str, List[Ion]],
    data_ions_per_block: Dict[str, List[Ion]],
    ion_mapping_per_block: Dict[str, Dict[int, Tuple[Ion, Tuple[int, int]]]],
    wise_config: QCCDWiseArch,
    k: int,
) -> Dict[str, np.ndarray]:
    """Run regularPartition + hillClimbOnArrangeClusters per block.
    
    Each block is mapped to its own allocated sub-grid region. Different
    blocks NEVER share traps.
    
    Returns per-block initial layout arrays (block-local coordinates).
    """
    layouts = {}
    for block_name, sub_grid in block_sub_grids.items():
        r0, c0, r1, c1 = sub_grid.grid_region
        block_rows = r1 - r0
        block_cols = c1 - c0
        
        # Partition THIS block's ions into clusters
        clusters = regularPartition(
            measurement_ions_per_block[block_name],
            data_ions_per_block[block_name],
            k,
            isWISEArch=True,
            maxClusters=block_rows * block_cols,
        )
        
        # Map clusters to positions within THIS block's sub-grid only
        block_grid_pos = [
            (c, r) for r in range(block_rows) for c in range(block_cols)
        ]
        grid_positions = hillClimbOnArrangeClusters(
            clusters, allGridPos=block_grid_pos
        )
        
        # Build block-local arrangement array
        layout = build_arrangement_from_positions(
            clusters, grid_positions, block_rows, block_cols, k
        )
        layouts[block_name] = layout
        
    return layouts
```

```
    Per-block mapping (disjoint sub-grids):
    ┌─────────────────┬─────────────────┐
    │     block_0     │     block_1     │
    │  D0  M1  D2  M3 │  D5  M6  D7  M8│  ← block_0 ions only in left region
    │  D4  M5  D6  M7 │  D9  M10 D11 M12│  ← block_1 ions only in right region
    └─────────────────┴─────────────────┘
    sub_grid: (0,0,2,4)  sub_grid: (0,4,2,4)
    
    Level 1 slicing now works:
    EC phase: extract (0,0,2,4) for block_0, extract (0,4,2,4) for block_1
    Gadget phase: concatenate both regions → merged 2-block grid
```

#### Sub-Grid Region Allocation

Sub-grid regions are allocated based on the gadget's `QubitAllocation` spatial arrangement. Blocks are placed side-by-side along the column axis (x-axis) with an optional gap column between them:

```python
def allocate_block_regions(
    block_names: List[str],
    qubits_per_block: Dict[str, int],
    k: int,
    gap_cols: int = 0,
) -> Dict[str, Tuple[int, int, int, int]]:
    """Allocate non-overlapping (r0, c0, r1, c1) regions for each block.
    
    Invariant: for all block pairs (A, B),
        region_A ∩ region_B == ∅  (no shared rows × columns)
    """
    regions = {}
    col_cursor = 0
    max_rows = 0
    
    for name in block_names:
        n_qubits = qubits_per_block[name]
        n_rows = int(math.ceil(math.sqrt(n_qubits)))
        n_cols = int(math.ceil(n_qubits / (n_rows * k)))
        max_rows = max(max_rows, n_rows)
        
        regions[name] = (0, col_cursor, n_rows, col_cursor + n_cols)
        col_cursor += n_cols + gap_cols
    
    return regions
```

#### Disjointness Assertions

Every code path that accesses block sub-grids MUST verify disjointness:

```python
def assert_disjoint_blocks(sub_grids: Dict[str, BlockSubGrid]) -> None:
    """Assert no two blocks share any trap position."""
    all_positions: Set[Tuple[int, int]] = set()
    for name, sg in sub_grids.items():
        r0, c0, r1, c1 = sg.grid_region
        block_positions = {(r, c) for r in range(r0, r1) for c in range(c0, c1)}
        overlap = all_positions & block_positions
        assert not overlap, (
            f"Block '{name}' overlaps with previously allocated blocks "
            f"at positions {overlap}"
        )
        all_positions |= block_positions

def assert_ions_in_own_region(
    ion_positions: Dict[int, Tuple[int, int]],
    sub_grids: Dict[str, BlockSubGrid],
) -> None:
    """Assert every ion is within its owning block's sub-grid region."""
    for name, sg in sub_grids.items():
        r0, c0, r1, c1 = sg.grid_region
        for ion_idx in sg.ion_indices:
            pos = ion_positions[ion_idx]
            assert r0 <= pos[0] < r1 and c0 <= pos[1] < c1, (
                f"Ion {ion_idx} of block '{name}' at position {pos} "
                f"is outside its sub-grid region ({r0},{c0})→({r1},{c1})"
            )
```

#### Impact on `build_topology`

The existing `WISEArchitecture.build_topology()` calls `regularPartition(measurement_ions, data_ions, ...)` with all ions. For gadget compilation, we **bypass** this and instead:

1. Call `map_qubits_per_block()` to get per-block layouts (as described above)
2. Build the `WISEArchitecture` with pre-assigned trap positions (skip `regularPartition` and `hillClimbOnArrangeClusters`)
3. Or equivalently, call `build_topology()` N times — once per block — each with only that block's ions and a sub-architecture sized to the block's region

This is analogous to how `_patch_and_route` creates a sub-grid per patch and solves it independently: each block becomes its own "super-patch" at Level 1.

### 3.2 Phase Decomposition

**Input**: `QECMetadata.phases` (built by `QECMetadata.from_gadget_experiment()`)

The decomposer iterates over `QECMetadata.phases` and emits `PhaseRoutingPlan` objects. This is a general loop — it handles any number of blocks and any alternation of EC and gadget phases.

**General structure** (from `QECMetadata.phases`):
```
for phase in metadata.phases:
    if phase.phase_type == "init":
        skip (no routing)
    elif phase.phase_type == "stabilizer_round_*":
        emit PhaseRoutingPlan(phase_type="ec",
            interacting_blocks=[],           # no cross-block gates
            idle_blocks=[],                  # all blocks route independently
            all_blocks=all_block_names,
            num_rounds=phase.num_rounds,
            is_cached=True,
            identical_to_phase=phase.identical_to_phase)
    elif phase.phase_type == "gadget":
        interacting = phase.active_blocks   # e.g. ["bell_a", "bell_b"]
        idle = [b for b in all_block_names if b not in interacting]
        emit PhaseRoutingPlan(phase_type="gadget",
            interacting_blocks=interacting,
            idle_blocks=idle,               # EXCLUDED from SAT grid
            num_rounds=1,
            is_cached=False)                # gadget phases are unique
        # Also emit transition plans at boundaries (see §3.5)
    elif phase.phase_type == "measure":
        skip (no routing)
```

**Examples**:

TransversalCNOT (2 blocks, 1 gadget phase):
```
Plan[0]: EC       all_blocks=[blk0, blk1]  interacting=[]  idle=[]  rounds=d  cached=True
Plan[1]: TRANS    EC→Gadget               merge blk0+blk1
Plan[2]: GADGET   interacting=[blk0,blk1]  idle=[]          rounds=1
Plan[3]: TRANS    Gadget→EC               split back to sub-grids
Plan[4]: EC       identical_to_phase=0      rounds=d  cached=True (replay)
```

KnillEC (3 blocks, 3 gadget phases):
```
Plan[0]:  EC       all_blocks=[data, bell_a, bell_b]  rounds=d  cached=True
Plan[1]:  TRANS    EC→Gadget2              merge bell_a+bell_b, EXCLUDE data
Plan[2]:  GADGET   interacting=[bell_a, bell_b]  idle=[data]   rounds=1
Plan[3]:  TRANS    Gadget2→EC              split bell_a, bell_b back to sub-grids
Plan[4]:  EC       identical_to_phase=0     rounds=d  cached=True (replay)
Plan[5]:  TRANS    EC→Gadget3              merge data+bell_a, EXCLUDE bell_b
Plan[6]:  GADGET   interacting=[data, bell_a]  idle=[bell_b]  rounds=1
Plan[7]:  TRANS    Gadget3→EC              split data, bell_a back
Plan[8]:  EC       identical_to_phase=0     rounds=d  cached=True (replay)
```

CSSSurgeryCNOT (3 blocks, 5 gadget phases):
```
Plan[0]:  EC       all_blocks=[blk0, blk1, blk2]  rounds=d  cached=True
Plan[1]:  TRANS    merge blk0+blk1, EXCLUDE blk2
Plan[2]:  GADGET   interacting=[blk0, blk1]  idle=[blk2]   (ZZ_merge)
Plan[3]:  TRANS    split back
Plan[4]:  EC       identical_to_phase=0 (replay)
Plan[5]:  GADGET   interacting=[blk0, blk1]  idle=[blk2]   (ZZ_split)
...
Plan[N]:  GADGET   interacting=[blk1, blk2]  idle=[blk0]   (XX_merge)
...etc
```

**Key invariant**: During any gadget phase, `len(interacting_blocks)` is always exactly 2 (all existing gadgets have pairwise interactions). Idle blocks are **not on the grid**.

**Idle block ion state**: When a block is idle during a gadget phase, its ions remain parked in the positions determined by the last EC round (which has ion-return BT, so they're at their canonical positions). No SAT variables, no reconfigurations, no grid cells allocated. When the next EC phase begins, we simply resume from those known positions.

### 3.3 MS Pair Derivation from QECMetadata

**For EC rounds**: Derive MS pairs from `QECMetadata.x_stabilizers.cnot_schedule` and `QECMetadata.z_stabilizers.cnot_schedule`:

```python
def derive_ms_pairs_from_metadata(
    qec_meta: QECMetadata,
    qubit_to_ion: Dict[int, int],
    block_name: Optional[str] = None,
) -> List[List[Tuple[int, int]]]:
    """Convert CNOT schedule from metadata to ion-index MS pairs.
    
    Returns one list-of-pairs per parallel MS round (4 rounds for
    rotated surface code: N, E, S, W CNOT layers → 4 MS rounds).
    """
    ms_rounds = []
    
    # Get stabilizer info (per-block if specified)
    x_sched = qec_meta.x_stabilizers.cnot_schedule or []
    z_sched = qec_meta.z_stabilizers.cnot_schedule or []
    
    # If block_name specified, filter to only qubits in that block
    if block_name:
        block_qubits = set()
        for ba in qec_meta.block_allocations:
            if ba.block_name == block_name:
                block_qubits.update(ba.data_qubits)
                block_qubits.update(ba.x_ancilla_qubits)
                block_qubits.update(ba.z_ancilla_qubits)
    
    for layer in x_sched + z_sched:
        pairs = []
        for ctrl, tgt in layer:
            if block_name and (ctrl not in block_qubits or tgt not in block_qubits):
                continue
            ion_ctrl = qubit_to_ion[ctrl]
            ion_tgt = qubit_to_ion[tgt]
            # Sort as (ancilla, data) — ancilla has label "M"
            pairs.append((ion_ctrl, ion_tgt))
        if pairs:
            ms_rounds.append(pairs)
    
    return ms_rounds
```

**For gadget rounds**: Extract cross-block MS pairs from the gadget's phase interaction pattern. Each gadget phase involves exactly 2 interacting blocks. The pairs come from the gadget's `emit_next_phase()` output or from the stim circuit segment for that phase.

```python
def derive_gadget_ms_pairs(
    gadget: Gadget,
    phase_index: int,
    qubit_allocation: QubitAllocation,
    interacting_blocks: List[str],
    qubit_to_ion: Dict[int, int],
) -> List[List[Tuple[int, int]]]:
    """Extract MS pairs for a specific gadget phase.
    
    Works for any 2-block interaction: TransversalCNOT, KnillEC phases,
    CSSSurgery merge/split phases, teleportation gadgets.
    
    Args:
        gadget: The gadget object
        phase_index: Which gadget phase (0-indexed)
        qubit_allocation: Full allocation with all blocks
        interacting_blocks: The 2 blocks that interact in this phase
        qubit_to_ion: Global qubit→ion mapping
    """
    block_a = qubit_allocation.blocks[interacting_blocks[0]]
    block_b = qubit_allocation.blocks[interacting_blocks[1]]
    
    # Get the CX/CZ pairs for this phase from the gadget
    # (implementation depends on gadget type — may use emit_next_phase()
    # or extract from stim circuit segment)
    phase_pairs = gadget.get_phase_pairs(phase_index)  # qubit-index pairs
    
    ms_rounds = []
    round_pairs = []
    for ctrl_q, tgt_q in phase_pairs:
        round_pairs.append((qubit_to_ion[ctrl_q], qubit_to_ion[tgt_q]))
    if round_pairs:
        ms_rounds.append(round_pairs)
    
    return ms_rounds
```

**Note**: The ion indices in the returned pairs are **renumbered to the merged 2-block grid**, not the full N-block grid. Idle block ions are not present.

### 3.4 Per-Block Independent Routing (EC Phases)

For each block during EC rounds:

1. **Extract sub-grid**: The block's `BlockSubGrid` defines the grid region.
2. **Build sub-grid arrangement**: `A_sub[n_rows_block × n_cols_block]` — ions of this block only.
3. **Derive MS pairs**: From `QECMetadata` CNOT schedules, filtered to this block's qubits, mapped to ion indices.
4. **Ion return constraint**: Add BT hard clauses pinning all active ions to their starting positions at the end of the round:
   ```python
   BT_return = [{ion: (start_row, start_col) for ion, (start_row, start_col) in ion_positions.items()}]
   ```
   This is passed as the BT parameter to `optimal_QMR_for_WISE` with `bt_soft=False` (hard clause).
5. **Solve once via `_patch_and_route()`**: This is the Level-2 spatial slicer. It tiles the block's sub-grid into overlapping constant-size patches (`subgrid_width × subgrid_height`, e.g. 4×3) and calls `optimal_QMR_for_WISE` on each patch. For a d=3 block (17 ions, ~5×4 grid), this may be 1-2 patches. For a d=12 block (287 ions, ~17×12 grid), this may be ~14 patches. The sub-grid is always small enough relative to the full N-block grid that Level-2 tiling is efficient.
6. **Cache result**: Store `(layouts, schedule, P_horizon)` keyed by `round_signature`.
7. **Replay**: For subsequent identical rounds, deepcopy the cached result and apply directly (no SAT call, no Level-2 patching).

**Parallelism**: Route all N blocks simultaneously in separate processes. Each gets `cpu_count // N_blocks` cores for inner SAT parallelism. For 3-block gadgets (KnillEC, CSSSurgery), this means 3-way parallel with 4 cores each on a 12-core machine.

### 3.4.1 Block Schedule Merging (`_merge_block_schedules`)

After independently routing N blocks during EC phases (§3.4), we hold N separate per-block schedules — each containing WISE odd-even sorter passes with H/V swap operations in **block-local coordinates**. These must be combined into a single schedule for the full architecture, analogous to how `_merge_patch_schedules` combines per-patch schedules at Level 2.

#### Relationship to `_merge_patch_schedules`

The existing Level 2 merge pipeline works as follows:

```
Per-patch schedules (from _patch_and_route)
    │
    ▼
_merge_patch_schedules(patch_schedules, R)        ← iterates per round
    │
    ▼
_merge_patch_round_schedules(round_scheds)        ← per round
    │
    ├── _split_patch_round_into_HV(...)           ← split H vs V passes
    ├── _merge_phase_passes(H_passes, "H")        ← merge H passes by parity
    ├── _merge_phase_passes(V_passes, "V")        ← merge V passes by parity
    └── concat: merged_H + merged_V               ← H-before-V ordering
```

Block schedule merging is a **Level 1 analogue** with identical structure but different semantics:

```
Per-block schedules (from route_blocks_parallel)
    │
    ▼
_merge_block_schedules(block_schedules, R)        ← iterates per round
    │
    ▼
_merge_block_round_schedules(round_scheds)        ← per round
    │
    ├── _remap_to_global(block_sched, sub_grid)   ← block-local → global coords
    ├── _split_patch_round_into_HV(...)           ← REUSE existing splitter
    ├── _merge_phase_passes(H_passes, "H")        ← REUSE existing merger
    ├── _merge_phase_passes(V_passes, "V")        ← REUSE existing merger
    └── concat: merged_H + merged_V               ← H-before-V ordering
```

#### Why Disjoint Sub-Grids Make This Safe

In `_merge_patch_schedules`, patches are non-overlapping tile regions within the same grid. The merge is safe because patches don't share trap columns (each patch has `c0:c1` disjoint from others in the same tiling offset). The same guarantee holds for blocks — but stronger:

- **Disjoint column ranges**: Block A occupies columns `[c0_A, c1_A)`, block B occupies `[c0_B, c1_B)`, and `c1_A ≤ c0_B` (with optional gap). This means H-swaps (within a row across columns) from different blocks can never touch the same ion pair. V-swaps (within a column across rows) from different blocks are in different column ranges.

- **No physical resource conflicts**: Because blocks are on disjoint sub-grids (§3.1.1), swap operations from different blocks can execute concurrently on the hardware without ion collisions or junction contention.

- **Parity compatibility**: The WISE odd-even comparator network requires that all swaps in a single pass share the same comparator parity (odd or even column/row index for the comparator). Since blocks occupy different column ranges, their swap operations naturally have well-defined parities that can be merged using the existing `_merge_phase_passes` parity-matching logic.

#### Coordinate Remapping

Each block's schedule uses block-local coordinates (rows/columns relative to `(0, 0)` of the block's sub-grid). Before merging, swap coordinates must be remapped to the global grid:

```python
def _remap_block_schedule_to_global(
    block_schedule: List[List[Dict[str, Any]]],
    sub_grid: BlockSubGrid,
) -> List[List[Dict[str, Any]]]:
    """Remap swap coordinates from block-local to global grid coordinates.
    
    H-swaps: (row, col_pair) → (row + r0, col_pair + c0)
    V-swaps: (col, row_pair) → (col + c0, row_pair + r0)
    """
    r0, c0, _, _ = sub_grid.grid_region
    remapped = []
    for round_passes in block_schedule:
        remapped_round = []
        for pass_info in round_passes:
            new_pass = {
                "phase": pass_info["phase"],
                "h_swaps": [
                    _offset_swap(s, row_offset=r0, col_offset=c0)
                    for s in pass_info.get("h_swaps", [])
                ],
                "v_swaps": [
                    _offset_swap(s, row_offset=r0, col_offset=c0)
                    for s in pass_info.get("v_swaps", [])
                ],
            }
            remapped_round.append(new_pass)
        remapped.append(remapped_round)
    return remapped
```

#### Full Block Merge Function

```python
def _merge_block_schedules(
    block_schedules: Dict[str, List[List[Dict[str, Any]]]],
    sub_grids: Dict[str, BlockSubGrid],
    R: int,
) -> List[List[Dict[str, Any]]]:
    """Merge independently-routed block schedules into a unified schedule.
    
    This is the Level 1 analogue of _merge_patch_schedules (Level 2).
    
    1. Remap each block's schedule from block-local to global coordinates
    2. For each round, split into H and V passes
    3. Merge H passes across all blocks (respecting odd-even parity)
    4. Merge V passes across all blocks (respecting odd-even parity)
    5. Concatenate: H passes first, then V passes
    
    Because blocks occupy disjoint sub-grid regions (§3.1.1), swap
    operations from different blocks never conflict physically. The
    merge only needs to respect the WISE sorter pass structure (H-before-V,
    same-parity passes combined).
    """
    # Step 1: remap all block schedules to global coordinates
    global_schedules = []
    for block_name, sched in block_schedules.items():
        remapped = _remap_block_schedule_to_global(sched, sub_grids[block_name])
        global_schedules.append(remapped)
    
    # Step 2: merge using the SAME merge logic as patch schedules
    # _merge_patch_schedules works on any list of per-region schedules
    # with non-overlapping swap coordinates
    merged = _merge_patch_schedules(global_schedules, R)
    
    return merged
```

**Key insight**: We **reuse** `_merge_patch_schedules` directly. It doesn't care whether its inputs came from Level 2 patches or Level 1 blocks — it only needs non-overlapping swap coordinates and the same H/V + parity structure, which both levels provide.

#### Three-Level Merge Hierarchy

For a full gadget EC round, the merge hierarchy is:

```
Level 2 (within each block):  _merge_patch_schedules
    │                          patches within block_0's sub-grid
    │                          patches within block_1's sub-grid
    │                          patches within block_2's sub-grid (KnillEC)
    ▼
Level 1 (across blocks):      _merge_block_schedules
    │                          block_0 schedule ← (already patch-merged)
    │                          block_1 schedule ← (already patch-merged)
    │                          block_2 schedule ← (already patch-merged)
    ▼
Unified schedule:              all H passes, then all V passes
                               for the full N-block architecture
```

Each level's merge is safe because regions at that level are disjoint. Level 2 patches within a block are disjoint by construction (checkerboard tiling). Level 1 blocks are disjoint by the invariant in §3.1.1.

#### Edge Case: Gadget Phases

During gadget phases, only the interacting blocks are merged onto a single grid. The gadget phase uses a **single** `_patch_and_route` call on this merged grid, which produces a single schedule. There is no block-level merge needed — the merge is handled entirely at Level 2 within `_patch_and_route`. The idle blocks have no schedule (they are excluded from the grid).

However, when assembling the final timeline (§3.6), the gadget phase schedule (covering interacting blocks) must be interleaved with the idle blocks' no-op. Since idle blocks have no operations during gadget phases, this is trivial — just include the gadget schedule as-is.

### 3.5 Transition Routing

Transitions occur at **every** EC ↔ gadget phase boundary. For a KnillEC gadget there are 4 transition points; for CSSSurgery there are up to 10.

**EC → Gadget transition** (before each gadget phase):

Only the *interacting* blocks' ions migrate from their sub-grids to the merged 2-block gadget grid. Idle blocks stay parked.

Strategy: Use BT pinning in the gadget solve.

1. **Compute target positions**: Determine where each interacting-block ion should be on the merged gadget grid. This comes from the mapper's initial layout for the 2-block merged grid.
2. **Solve gadget round with BT**: The gadget SAT solve receives BT pins from the last EC round's final layout (mapped to merged grid coordinates). The SAT solver handles the migration as part of its normal reconfiguration:
   ```
   Last EC layout (sub-grid coords) → [map to merged grid] → [BT-guided reconfig] → Gadget-optimal layout → [MS gates]
   ```
3. **Idle blocks**: Do nothing. Their ions remain in their last EC positions (canonical, due to ion-return BT).

**Gadget → EC transition** (after each gadget phase):

Interacting blocks' ions return from the merged gadget grid to their per-block sub-grids.

Strategy: BT-pin the last gadget round to the cached EC initial positions.

1. **Target**: The `initial_layout` from the cached EC round solution for each interacting block.
2. **Implementation**: Pass `BT = [{ion: target_pos for all interacting ions}]` as hard constraints on the gadget's final layout. The gadget solve optimises the gate while ensuring ions end up at their EC-compatible positions.

If a single-round gadget solve with BT pins is infeasible (SAT returns UNSAT), fall back to:
1. Solve gadget without BT pins (free placement)
2. Insert a dedicated transition round: solve a reconfig-only SAT (no MS gates, just minimise reconfiguration cost from gadget-final to EC-initial layout)

**Transition cost accounting**: Each transition adds at most one additional SAT call (for the dedicated reconfig fallback). With BT-embedded transitions, the cost is zero additional calls. For a KnillEC with 2 gadget phases that require reconfig, the total is at most 2 extra transition calls.

### 3.6 Routing Sequence Assembly

After all phases are routed, assemble the full operation sequence by iterating over `PhaseRoutingPlan` objects in order:

```python
all_ops = []
all_barriers = []
ec_cache = {}  # block_name -> CachedRoundResult

for plan in phase_routing_plans:
    if plan.phase_type == "ec":
        # Each block routes independently on its sub-grid
        for block_name in plan.all_blocks:
            if plan.identical_to_phase is not None:
                # Replay from cache
                cached = ec_cache[block_name]
            else:
                cached = route_and_cache(block_name, ...)
                ec_cache[block_name] = cached
            
            for round_idx in range(plan.num_rounds):
                ops, barriers = replay_cached_round(cached, round_idx)
                all_ops.extend(ops)
                all_barriers.extend(barriers)
    
    elif plan.phase_type == "gadget":
        # Level 1: Build merged grid with ONLY interacting blocks
        merged_grid = build_phase_grid(
            [sub_grids[b] for b in plan.interacting_blocks], k)
        
        # Idle blocks: no-op (ions parked at EC-final positions)
        # Their ions are NOT in the SAT variable space
        assert len(plan.idle_blocks) == len(plan.all_blocks) - len(plan.interacting_blocks)
        
        # Level 2: _patch_and_route tiles merged_grid into sub-patches
        gadget_result = route_gadget_phase(
            merged_grid, plan.interacting_blocks,
            plan.ms_pairs_per_round,
            subgridsize=(subgrid_width, subgrid_height, subgrid_increment),
            entry_bt=get_ec_final_layout(plan.interacting_blocks),
            exit_bt=get_ec_initial_layout(plan.interacting_blocks),
        )
        ops, barriers = apply_routing(gadget_result)
        all_ops.extend(ops)
        all_barriers.extend(barriers)
    
    elif plan.phase_type == "transition":
        # Handled embedded in gadget solve via BT (see §3.5)
        # Only explicit if BT-embedded solve was UNSAT
        if has_explicit_transition(plan):
            ops = build_transition_ops(...)
            all_ops.extend(ops)
```

This loop naturally handles any number of gadget phases with any number of blocks, including:
- TransversalCNOT: 1 iteration (EC → gadget → EC)
- KnillEC: 3 iterations (EC → gadget2 → EC → gadget3 → EC)
- CSSSurgery: 5 iterations (EC → ZZmerge → EC → ZZsplit → EC → XXmerge → EC → XXsplit → EC → ancMX)

### 3.7 Stim Circuit and Noise Injection

**Stim circuit**: Generated by `FaultTolerantGadgetExperiment.to_stim()`. This produces the complete ideal circuit with correct DETECTOR/OBSERVABLE_INCLUDE annotations. The routing optimisation does NOT modify the stim circuit — it only determines the physical operation schedule.

**Noise injection**: Uses the existing `TrappedIonExperiment` path:
```python
experiment = TrappedIonExperiment(
    code=code,
    gadget=gadget,
    architecture=arch,
    compiler=compiler,
    hardware_noise=noise,
    rounds=distance,
    basis='z',
)
ideal_circ = experiment.build_ideal_circuit()
experiment._compiled = compiled  # inject our optimised compiled result
noisy_circuit = experiment.apply_hardware_noise()
```

The `apply_hardware_noise()` method builds an `ExecutionPlan` from the `CompiledCircuit` and injects noise based on the physical operations and their durations.

---

## 6. Key Risk Areas & Mitigations

### Risk 1: BT Ion-Return Makes SAT UNSAT
The ion-return hard BT constraint adds `n_ions × 2` hard clauses (row + col pinning). For tight grids, this may make the SAT problem infeasible.

**Mitigation**: 
- Start with BT as hard clause (user preference)
- Monitor SAT return codes; if UNSAT, log a warning and retry with loose BT (soft clause with high weight)
- For very tight grids, fall back to verify-and-retry: solve without BT, check if ions returned, re-solve with BT only if they didn't

### Risk 2: Multi-Block QUBIT_COORDS Parity
The `int(float(x)) % 2` heuristic for data/measurement classification breaks with block offsets.

**Mitigation**: Always pass `data_qubit_idxs` from `QubitAllocation` to the compiler. Never rely on coordinate parity for multi-block circuits.

### Risk 3: Transition Routing Cost
Migrating ions between sub-grid and full-grid layouts may be expensive (many reconfig passes).

**Mitigation**:
- Use BT on the last gadget round so the gadget solver handles the transition within its own reconfig budget
- If BT-pinned gadget solve is much worse than free solve, accept the free solve and insert a dedicated reconfig-only transition round
- Optimise block placement on the grid to minimise transition distance (blocks side-by-side with minimal gap)

### Risk 4: Cached Routing Replay Correctness
Replaying cached routing assumes identical ion starting positions. If the ion-return BT doesn't achieve exact return (e.g. due to soft clauses), the replay will produce incorrect results.

**Mitigation**: 
- Use hard BT clauses (user confirmed)
- Add assertion: `np.array_equal(layout[-1], layout[0])` after each cached round solve
- If assertion fails, fall back to non-cached routing for this block

### Risk 5: Scheduling Independence and Block Schedule Merging
Treating block schedules as independent during EC phases assumes no cross-block resource contention (e.g. shared junction nodes on the grid). The merged schedule must correctly combine passes from different blocks.

**Mitigation**: 
- Sub-grids are non-overlapping by the disjoint block layout invariant (§3.1.1) → no physical resource conflicts
- `_merge_block_schedules` (§3.4.1) remaps block-local coordinates to global, then reuses `_merge_patch_schedules` which enforces H-before-V ordering and same-parity-per-pass correctness
- Disjoint column ranges guarantee swap operations from different blocks never touch the same ion pair
- Gap column between adjacent blocks prevents junction contention at block boundaries
- Integration tests (`test_block_schedule_merge_2blk`, `test_block_schedule_merge_3blk`) verify merge correctness

### Risk 6: Multi-Phase Ion State Consistency
With multiple gadget phases (KnillEC, CSSSurgery), ions alternate between sub-grid and merged-grid layouts multiple times. Accumulated position drift or BT failures could corrupt state.

**Mitigation**:
- Ion-return BT hard clauses guarantee exact return to canonical positions after every EC round
- Each gadget phase starts from known canonical positions (post-EC)
- Assert `layout[-1] == canonical_layout` at every phase boundary
- If assertion fails at any point, abort and fall back to un-optimised routing for remaining phases

### Risk 7: Idle Block Ion Leakage During Gadget SAT
When idle blocks are excluded from the SAT grid, the solver is unaware of their ions. If the merged gadget grid physically overlaps with where idle ions are "parked", there could be physical conflicts.

**Mitigation**:
- Sub-grids are non-overlapping by the disjoint block layout invariant (§3.1.1) — enforced by `assert_disjoint_blocks()` and `assert_ions_in_own_region()` at startup
- The merged gadget grid occupies only the grid regions of the interacting blocks
- Idle block ions are parked in their own sub-grid region, which is disjoint by construction
- Add check: `assert merged_grid_region ∩ idle_grid_region == ∅`
- Per-block `map_qubits_per_block()` guarantees ions are placed only within their block's allocated region — no ion can leak into another block's region

### Risk 8: Per-Block Mapping Quality vs Global Mapping
Running `regularPartition` + `hillClimbOnArrangeClusters` per block (within each block's sub-grid) may produce slightly worse initial layouts than a global mapping, because the global mapper can consider cross-block cluster proximity.

**Mitigation**:
- For EC phases, per-block mapping quality is all that matters (blocks route independently)
- For gadget phases, the initial layout on the merged grid is determined by the per-block layouts; the SAT solver handles migration via BT-guided reconfiguration
- The loss in initial layout quality is far outweighed by the exponential gain from Level 1 spatial slicing
- Monitor gadget phase reconfiguration cost: if excessive, consider a dedicated transition round to migrate ions to gadget-optimal positions before the SAT solve

---

## §10 Implementation Gaps — Fix Plan (Rev 3)

Five critical gaps between the spec and the current implementation.
Each is tracked with a checkbox — tick off as completed.

### Issue 4 — Restore Epoch-Aware Drain (Section 4b) `[x]`

**Problem:** The current `_drain_single_qubit_ops` helper in
`qccd_WISE_ion_route.py` is a simple epoch-blind greedy drain.
The committed (working) version had:

- **C1** Epoch ceiling: never pull ops from epochs later than the
  earliest remaining multi-qubit (MS) gate.
- **C2** Blocked-ions scan: once a 2Q gate is seen on an ion in
  `operationsLeft`, all later 1Q ops on that ion are ineligible
  this round.
- **Per-epoch, shortest-gate-first drain:** group by epoch →
  per-ion deques → drain fastest type first
  (XRot → YRot → Reset → Measure).

**Files changed:**

| File | Change |
|------|--------|
| `qccd_WISE_ion_route.py` `_drain_single_qubit_ops` | Replace ~30-line epoch-blind drain with the committed ~100-line epoch-aware drain |

**Logic to restore (from committed `ionRoutingWISEArch` §4b):**

```
1. Compute min_ms_epoch from remaining operationsLeft
2. Blocked-ions scan: iterate operationsLeft, block ions with 2Q gates
3. Filter eligible: 1Q ops with epoch <= min_ms_epoch, unblocked, valid trap
4. Group eligible by epoch → per-ion deques
5. Per epoch (ascending), drain by type order (XRot→YRot→Reset→Measure)
6. Per type group, greedy disjoint-ion execution with sub-barriers
```

---

### Issue 2 — Level 2 Spatial Slicing for Inactive Blocks `[x]`

**Problem:** `gadget_routing.py` line 1694:
`merged_sgs = (m_total_cols, m_n_rows, 0)` forces `increment=0`,
disabling Level 2 patch decomposition for gadget phases.

Level 1 already strips inactive blocks.  The merged grid only
contains active (interacting) blocks.  The SAT solver should use
normal patch decomposition on this reduced grid.

**Files changed:**

| File | Change |
|------|--------|
| `gadget_routing.py` line 1694 | `merged_sgs = subgridsize` (pass caller's subgridsize) |

---

### Issue 3 — Cross-Block Parallel Reconfiguration `[x]`

**Problem:** `_merge_disjoint_block_schedules` serialises passes
when phase labels differ at the same index.  Since blocks occupy
disjoint grid regions, the odd-even constraint only matters within
one column range — phase labels are cosmetic for correctness.

Reconfigurations, MS gates, and rotations from different blocks
must execute in parallel (the 1st reconfig in block 1 in parallel
with the 1st reconfig in block 2).

**Files changed:**

| File | Change |
|------|--------|
| `qccd_WISE_ion_route.py` `_merge_disjoint_block_schedules` | Always merge h_swaps+v_swaps regardless of phase label; add disjointness assertion |
| `gadget_routing.py` EC per-block routing | Use caller's `subgridsize` for large blocks instead of forcing `(cols*k, rows, 0)` |

---

### Issue 1 — QECMetadata as Single Source of Truth `[x]`

**Problem:** `ionRoutingGadgetArch` §1b derives `phase_pair_counts`
via fragile tick-epoch heuristics (~170 lines) that can silently
fall back to flat routing.  The routing code should NOT derive
epochs, topology, block grid regions, block assignments, or phase
scheduling.

**Files changed:**

| File | Change |
|------|--------|
| `pipeline.py` `PhaseInfo` | Add `cx_per_round: int = 0` and `ms_pair_count: int = 0` |
| `pipeline.py` `QECMetadata` | Add `cx_per_ec_round: Optional[int] = None` |
| `pipeline.py` `from_css_memory` | Compute `cx_per_ec_round` from CNOT layers |
| `pipeline.py` `from_gadget_experiment` | Compute `cx_per_ec_round`, populate per-phase `cx_per_round` and `ms_pair_count` |
| `qccd_WISE_ion_route.py` `ionRoutingGadgetArch` | Delete §1b epoch analysis (~170 lines); build `phase_pair_counts` from `[p.ms_pair_count for p in phases]`; read `cx_per_ec_round` from metadata |

---

### Issue 5 — Eliminate Double Routing `[x]`

**Problem:** `run_single_gadget_config` routes the circuit twice:
1. `route_full_experiment()` → phase-aware routing for timing
2. `_route_and_simulate()` → full monolithic routing + noise + decode

The second call is redundant and wastes ~50% of total time.

**Files changed:**

| File | Change |
|------|--------|
| `best_effort_compilation_WISE.py` `_route_and_simulate` | Add optional `qec_metadata`, `qubit_allocation`, `block_sub_grids` params; forward to `compiler.routing_kwargs` |
| `best_effort_compilation_WISE.py` `run_single_gadget_config` | Remove `route_full_experiment()` call; pass metadata through to single `_route_and_simulate()` |
| `trapped_ion_compiler.py` | Verify `qec_metadata`/`qubit_allocation`/`block_sub_grids` forwarded from `routing_kwargs` to `ionRoutingGadgetArch` |

---

### Execution Order

```
Issue 4 (drain)  → standalone, no deps, highest confidence
Issue 2 (L2)     → one-line fix, low risk
Issue 3 (merge)  → targeted fix in merge function
Issue 1 (meta)   → larger refactor, well-defined
Issue 5 (double) → depends on Issue 1
```

### Risk Assessment

| Issue | Risk | Mitigation |
|-------|------|------------|
| 4 | Low | Direct port of committed code |
| 2 | Low | Simple parameter passthrough |
| 3 | Low | Disjoint regions guarantee no conflicts; add assertions |
| 1 | Medium | Metadata factory computes from CNOT schedule; fast-fail if absent |
| 5 | Medium | Incremental: pass kwargs first, then remove redundant call |

---