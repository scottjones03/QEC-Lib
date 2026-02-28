# Gadget Metadata Fix Plan — Circuit-Counted Phase Pair Counts

## Problem

The routing layer (`ionRoutingGadgetArch`) needs per-phase `ms_pair_count`
values that exactly equal the number of CX instruction objects in each section
of the stim circuit.  The current metadata computation in
`QECMetadata.from_gadget_experiment()` uses **formulas** to predict these
counts, but the formulas are wrong for CSS Surgery (and would be wrong for any
gadget that interleaves EC rounds inside its phases):

| Phase | Formula result | Actual CX instructions | Root cause |
|-------|---------------|----------------------|------------|
| EC pre | `cx_per_round × rounds_before = 8` | **0** | CSS Surgery skips ALL blocks pre-gadget |
| Gadget ZZ merge | `sum(1 for rp if rp) = 2` | **28** | Only counts bridge rounds, not interleaved EC |
| Gadget ZZ split | 0 | 0 | OK |
| Gadget XX merge | 2 | **28** | Same as ZZ merge |
| Gadget XX split | 0 | 0 | OK |
| Gadget Anc MX | 0 | 0 | OK |
| EC post | `cx_per_round × rounds_after = 8` | **8** | Happens to be correct |
| **Total** | **20** | **64** | **Mismatch → fallback → SAT failure** |

The routing layer's sanity check `sum(phase_pair_counts) != len(parallelPairs)`
fails (20 ≠ 64), causing fallback to flat routing, which then fails with
`RuntimeError: SAT schedule did not realise target layout`.

## Root Cause

The formulas in `from_gadget_experiment()` were designed for TransversalCNOT
(single gadget phase, no interleaved EC, no skipped blocks).  They cannot
handle:

1. **Skipped EC blocks**: `get_blocks_to_skip_pre_rounds()` returns all blocks
   for CSS Surgery — pre-gadget EC emits 0 CX instructions, not 8.
2. **Interleaved EC inside gadget phases**: CSS Surgery merge phases call
   `builder.emit_round()` for each block within each merge round.  This adds
   `n_blocks × cx_per_ec_round` CX instructions per merge round that the
   formula ignores.
3. **Per-block EC emission in gadget phases**: During merge phases, each block's
   EC is emitted as separate CX instructions (not shared across blocks like in
   pre/post EC via `emit_scheduled_rounds()`).

## Design Principle

> **The stim circuit is the single source of truth.**
>
> The metadata's `ms_pair_count` must exactly match the number of CX
> instruction objects the compiler will find when it scans the circuit.
> Instead of predicting counts from formulas, **count them from the actual
> built circuit**.

## Implementation

### Step 1: Add CX counting helper

Add a helper function that counts CX/CNOT/CZ instruction objects in a
stim circuit (handling REPEAT blocks recursively):

```python
def _count_cx_instructions(circuit: stim.Circuit) -> int:
    n = 0
    for inst in circuit:
        if isinstance(inst, stim.CircuitRepeatBlock):
            n += inst.repeat_count * _count_cx_instructions(inst.body_copy())
        elif inst.name in ("CX", "CZ", "XCZ", "ZCX", "ZCZ"):
            n += 1
    return n
```

**Location**: `ft_gadget_experiment.py` (module-level helper).

### Step 2: Instrument `to_stim()` to count CX per section

In `FaultTolerantGadgetExperiment.to_stim()`, count CX instructions before
and after each circuit section:

```python
# After Phase 2.5 (prepare states), before Phase 3:
cx_before_pre = _count_cx_instructions(circuit)

# Phase 3: Pre-gadget EC
self._emit_pre_gadget_memory(circuit, builders)
cx_after_pre = _count_cx_instructions(circuit)

# Phase 4: Gadget phases (with per-phase tracking)
# ... (see Step 3 below)
cx_after_gadget = _count_cx_instructions(circuit)

# Phase 5: Post-gadget EC
self._emit_post_gadget_memory(...)
cx_after_post = _count_cx_instructions(circuit)
```

### Step 3: Per-gadget-phase CX counting via PhaseOrchestrator

Modify `PhaseOrchestrator.execute_phases()` to track CX instructions emitted
by each gadget phase.  Before/after each `gadget.emit_next_phase()` call,
count CX instructions and store the delta.

The per-phase counts are returned in `PhaseExecutionResult` as a new field
`phase_cx_counts: List[int]`.

### Step 4: Patch metadata with actual counts

After circuit construction in `to_stim()`, patch the pre-computed metadata
with actual CX counts:

```python
# Patch ec_pre
ec_pre_phase = next(p for p in self._qec_metadata.phases
                    if p.phase_type == 'stabilizer_round_pre')
ec_pre_phase.ms_pair_count = cx_after_pre - cx_before_pre

# Patch gadget phases
gadget_phases = [p for p in self._qec_metadata.phases
                 if p.phase_type == 'gadget']
for gp, cx_count in zip(gadget_phases, phase_result.phase_cx_counts):
    gp.ms_pair_count = cx_count

# Patch ec_post
ec_post_phase = next(p for p in self._qec_metadata.phases
                     if p.phase_type == 'stabilizer_round_post')
ec_post_phase.ms_pair_count = cx_after_post - cx_after_gadget
```

### Step 5: Simplify `ionRoutingGadgetArch` metadata reading

With accurate metadata, the routing layer no longer needs the
`_use_per_gadget_counts` branching or the even-distribution heuristic.
Simplify to:

```python
phase_pair_counts = [getattr(p, 'ms_pair_count', 0) for p in phases]
```

The sanity check `sum(phase_pair_counts) == len(parallelPairs)` should now
always pass for any gadget.

## Why Not Epoch-Based?

The previous attempt used `_tick_epoch` gaps on compiled `toMoveOps` entries
to segment `parallelPairs` into phases.  This failed because:

- Within-phase epoch gaps (3, between block EC layers) are close to
  between-phase gaps (4–5), making threshold-based segmentation fragile.
- It requires the routing layer to reverse-engineer phase structure from
  compiled output — violating the principle that metadata should be the
  single source of truth.

## Affected Files

| File | Change |
|------|--------|
| `ft_gadget_experiment.py` | Add `_count_cx_instructions()`, instrument `to_stim()` |
| `phase_orchestrator.py` | Add `phase_cx_counts` to `PhaseExecutionResult`, track per-phase CX |
| `pipeline.py` | Keep initial formula-based estimates (harmless, overwritten by patch) |
| `qccd_WISE_ion_route.py` | Simplify metadata reading (remove heuristic branches) |

## Test Plan

1. Run integration tests: `PYTHONPATH=src pytest src/.../demo/test_integration_layer.py -v`
2. Run E2E tests: `PYTHONPATH=src pytest src/.../demo/test_e2e.py -v`
3. Re-run notebook cell 24 (CSS Surgery) — should compile and animate
4. Re-run notebook cell 22 (TransversalCNOT) — regression check
