# Gadget Generalisation Plan

**Date**: 2026-02-22  
**Status**: Implementation in progress  
**Goal**: Make the trapped-ion routing pipeline generic over all gadgets in the repo.

---

## 1  Problem Statement

The routing pipeline (`gadget_routing.py`, `run.py`) was built around
`TransversalCNOTGadget` — a 2-block, 1-phase gadget.  It crashes on
`CSSSurgeryCNOTGadget` because:

1. **`derive_gadget_ms_pairs` hard-codes `len(interacting_blocks) == 2`.**  
   CSS Surgery phase 4 has `active_blocks = ["block_1"]` (single-block
   ancilla MX), which triggers `ValueError`.

2. **No gadget implements `get_phase_pairs()`,** so the fallback path
   (transversal data-qubit pairing) is always taken.  This is wrong for
   surgery (bridge qubits), teleportation (prep/meas phases), and KnillEC
   (rotating block pairs).

3. **`partition_grid_for_blocks` uses `allocate_block_regions` which lays
   blocks side-by-side along columns.**  This works for 2 blocks but
   ignores the gadget's `compute_layout()` geometry.  CSS Surgery
   specifies edge-adjacent blocks (block_0 right↔block_1 left,
   block_1 bottom↔block_2 top) — the grid mapper must follow this.

---

## 2  Gadget Inventory

| Gadget | Blocks | Phases | Single-Block? | Bridge Qubits? | Multi-Round? |
|--------|--------|--------|---------------|----------------|-----|
| `TransversalCNOT` | 2 | 1 | No | No | No |
| `CZHTeleport` | 2 | 3 | Yes (ph 0, 2) | No | No |
| `CNOTHTeleport` | 2 | 3 | Yes (ph 0, 2) | No | No |
| `KnillEC` | 3 | 3 | No (2-block, rotating pairs) | No | No |
| `CSSSurgeryCNOT` | 3 | 5 | Yes (ph 4) | Yes (ZZ + XX) | Yes (merge) |

Block layout patterns from `compute_layout()`:

| Gadget | Layout Geometry |
|--------|----------------|
| TransversalCNOT | `block_0` at origin, `block_1` auto-offset right |
| CZHTeleport | `data_block` at origin, `ancilla_block` auto-offset right |
| CNOTHTeleport | `data_block` at origin, `ancilla_block` auto-offset right |
| KnillEC | `data_block` at origin → `bell_a` right → `bell_b` right |
| CSSSurgeryCNOT | `block_0` at origin, `block_1` right-adjacent, `block_2` below block_1 |

---

## 3  Design

### 3.1  Add `get_phase_pairs()` to `Gadget` base class

```python
def get_phase_pairs(
    self, phase_index: int, alloc: QubitAllocation,
) -> List[List[Tuple[int, int]]]:
    """Return MS (2Q gate) pairs per round for this gadget phase.

    Returns List[List[(ctrl_qubit, tgt_qubit)]]:
        outer list = rounds within the phase,
        inner list = parallel pairs within each round.
    Returns [[]] or [] for phases with no 2Q interactions
    (preparation, measurement, single-block).
    """
```

Each gadget implements this using its knowledge of which qubits interact,
reading from `alloc.blocks[name].get_data_qubits()` and
`alloc.bridge_ancillas`.  The routing layer becomes a generic consumer.

**Implementation per gadget:**

| Gadget | Phase | Returns |
|--------|-------|---------|
| TransversalCNOT | 0 (CNOT) | `[[(ctrl_data[i], tgt_data[i]) for i]]` |
| CZHTeleport | 0 (prep) | `[]` |
| CZHTeleport | 1 (CZ) | `[[(data[i], anc[i]) for i]]` |
| CZHTeleport | 2 (meas) | `[]` |
| CNOTHTeleport | 0 (prep) | `[]` |
| CNOTHTeleport | 1 (CNOT) | `[[(data[i], anc[i]) for i]]` |
| CNOTHTeleport | 2 (meas) | `[]` |
| KnillEC | 0 (prep) | `[]` |
| KnillEC | 1 (bell CNOT) | `[[(bell_a[i], bell_b[i]) for i]]` |
| KnillEC | 2 (bell meas) | `[[(data[i], bell_a[i]) for i]]` |
| CSSSurgeryCNOT | 0 (ZZ merge) | `d` rounds: `[[(ctrl_data[s], bridge_zz[b]) ...] for rnd]` |
| CSSSurgeryCNOT | 1 (ZZ split) | `[]` (destructive M only) |
| CSSSurgeryCNOT | 2 (XX merge) | `d` rounds: `[[(bridge_xx[b], tgt_data[s]) ...] for rnd]` |
| CSSSurgeryCNOT | 3 (XX split) | `[]` (destructive MX only) |
| CSSSurgeryCNOT | 4 (anc MX) | `[]` (single-block M only) |

### 3.2  Handle empty-MS-pair gadget phases

`decompose_into_phases` currently requires every gadget phase to produce
MS pairs.  Phases returning `[]` (prep, measurement, single-block) are
**no-MS phases** — the routing layer should:

- Still emit them as `PhaseRoutingPlan(phase_type="gadget", ms_pairs_per_round=[])`
- When routing encounters an empty-MS gadget phase, skip SAT routing
  entirely — emit a no-op step to maintain phase ordering in the schedule.

### 3.3  Multi-round gadget phases (CSS Surgery merge)

**Approach: flatten into the `ms_pairs_per_round` list.**

`get_phase_pairs()` for merge phases returns `d` rounds directly.  Each
round represents one bridge parity check + interleaved EC.  The routing
layer receives them as multiple MS rounds to route within a single phase
(same as it handles multi-round EC phases).

This is transparent to the router — it already handles
`ms_pairs_per_round` as a list of rounds via `_route_round_sequence`.

### 3.4  Follow gadget `compute_layout()` geometry for grid partitioning

**Replace `allocate_block_regions`'s side-by-side-columns layout** with a
layout derived from the gadget's actual `compute_layout()` block offsets.

Flow:
1. Call `gadget.compute_layout(codes)` → `GadgetLayout`
2. Read block offsets from `layout.blocks[name].offset`
3. Map offsets to grid regions: normalise to grid coordinates, maintaining
   relative adjacency
4. Fall back to side-by-side if no layout available

This ensures CSS Surgery gets block_0 at origin, block_1 right of
block_0, block_2 below block_1 — matching the logical layout.

### 3.5  Plug changes into `decompose_into_phases`

Pass `gadget` and `qubit_allocation` through so `decompose_into_phases`
can call `gadget.get_phase_pairs(phase_index, alloc)` directly instead of
the fallback `derive_gadget_ms_pairs`.

### 3.6  Update `compile_gadget_for_animation` in `run.py`

Ensure the animation pipeline passes `qubit_allocation` through and uses
the layout-derived grid partitioning for 3-block gadgets.

---

## 4  Files Changed

| File | Changes |
|------|---------|
| `src/qectostim/gadgets/base.py` | Add `get_phase_pairs()` default + docstring |
| `src/qectostim/gadgets/transversal_cnot.py` | Implement `get_phase_pairs()` |
| `src/qectostim/gadgets/css_surgery_cnot.py` | Implement `get_phase_pairs()` |
| `src/qectostim/gadgets/knill_ec.py` | Implement `get_phase_pairs()` |
| `src/qectostim/gadgets/teleportation_h_gadgets.py` | Implement `get_phase_pairs()` for both gadgets |
| `src/qectostim/experiments/hardware_simulation/trapped_ion/utils/gadget_routing.py` | Fix `derive_gadget_ms_pairs`, `decompose_into_phases`, `partition_grid_for_blocks` |
| `src/qectostim/experiments/hardware_simulation/trapped_ion/demo/run.py` | Adapt `compile_gadget_for_animation` for N-block |

---

## 5  Testing

1. Unit: `get_phase_pairs()` returns correct pairs for each gadget
2. Integration: `decompose_into_phases` succeeds for all 5 gadgets
3. E2E: Notebook cell 24 (CSSSurgeryCNOT) compiles + animates
4. Regression: Existing TransversalCNOT tests still pass (30/30 + 20/20)
