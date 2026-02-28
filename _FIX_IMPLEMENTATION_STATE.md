# Fix Implementation State — CRITICAL CONTEXT

## Files Being Modified
1. `src/qectostim/experiments/hardware_simulation/trapped_ion/compiler/qccd_WISE_ion_route.py`
2. `src/qectostim/experiments/hardware_simulation/trapped_ion/utils/gadget_routing.py`

## Fix A: DONE ✅
- qccd_WISE_ion_route.py: Moved `barriers.append(len(allOps))` from inside `while grp_remaining` loop to after type group completes
- Location: line ~2218 in `_drain_single_qubit_ops` function
- One barrier per type group instead of per greedy pass

## Fix B: PARTIALLY DONE
### Part 1: DONE ✅
- gadget_routing.py in `decompose_into_phases` (~line 1565): Added round_signature computation for gadget phases
- Computes signature from sorted ms_pairs, checks seen_signatures for dedup

### Part 2: IN PROGRESS - CRITICAL ISSUE
The `if not _gadget_cache_hit:` guard at line 3086 is followed by UN-INDENTED code.
ALL code from line 3087 (`interacting_names = plan.interacting_blocks or all_block_names`)  
through line ~3323 (`phase_steps, current_layout = _apply_post_gadget_transition(...)`)
must be INDENTED one extra level (4 spaces) under `if not _gadget_cache_hit:`.

This is a ~240 line block that needs indenting. The block has:
- Lines 3087-3092: interacting_names setup
- Lines 3094-3100: _use_level1_slicing computation
- Lines 3101-3123: _use_bbox computation  
- Lines 3124-3295: L1 slicing path (if _use_level1_slicing and interacting_sgs)
- Lines 3296-3323: Full-grid routing path (else)

After the indented block, need to add:
```python
                # Fix B: Cache fresh gadget routing for future phases
                if cache_ec_rounds and plan.round_signature is not None:
                    ec_cache[plan.round_signature] = (
                        _gadget_starting_layout, phase_steps,
                    )
```
And `_gadget_starting_layout = np.array(current_layout, copy=True)` needs to be captured BEFORE routing starts.

## Fix C: NOT STARTED
- gadget_routing.py in `decompose_into_phases`, the interleaving logic (~lines 1530-1555)
- Currently: for each gadget round, extends _interleaved with bridge sub-rounds, THEN extends with ALL ec_pairs_all
- Fix: Merge bridge sub-round[i] with ec_pairs_all[i] into combined rounds where ions are disjoint
- Reduces rounds per merge cycle from n_sub + n_ec to max(n_sub, n_ec)

### Current interleaving code (lines ~1530-1555):
```python
if ec_pairs_all:
    _interleaved: List[List[Tuple[int, int]]] = []
    for _gadget_round in ms_pairs:
        _sub_rounds = _split_shared_ion_rounds([_gadget_round])
        _interleaved.extend(_sub_rounds)
        _round_labels.extend(["bridge"] * len(_sub_rounds))
        _interleaved.extend(ec_pairs_all)
        _round_labels.extend(["ec"] * len(ec_pairs_all))
    ms_pairs = _interleaved
```

### Fix C replacement:
```python
if ec_pairs_all:
    _interleaved: List[List[Tuple[int, int]]] = []
    for _gadget_round in ms_pairs:
        _sub_rounds = _split_shared_ion_rounds([_gadget_round])
        # Fix C: Merge bridge sub-rounds with EC rounds into
        # combined routing rounds.  Bridge pairs are inter-block
        # and EC pairs are intra-block, so they typically operate
        # on disjoint ions and can share a routing round.
        n_sub = len(_sub_rounds)
        n_ec = len(ec_pairs_all)
        n_combined = max(n_sub, n_ec)
        for _ci in range(n_combined):
            combined = []
            _label = "ec"
            if _ci < n_sub:
                combined.extend(_sub_rounds[_ci])
                _label = "bridge" if _ci >= n_ec else "combined"
            if _ci < n_ec:
                combined.extend(ec_pairs_all[_ci])
            if combined:
                _interleaved.append(combined)
                _round_labels.append(_label)
    ms_pairs = _interleaved
```

## After all fixes: need syntax check + verification test
- Syntax: `python -c "import ast; ast.parse(open('FILE').read()); print('OK')"`
- Verification: run `_test_fix13_fast.py` or similar against CSS Surgery d=2
