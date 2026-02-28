# Parallel Stabilizer Emission Fix — Spec & Implementation Plan

## 1. Problem Statement

The gadget compilation animation shows:
1. **Intra-block serialization**: MS gates for X and Z stabilizers within a
   single code block execute serially (8 routing rounds per EC round)
   instead of in parallel (4 interleaved rounds), doubling reconfigurations.
2. **Inter-block serialization**: Stabilizer checks across different blocks
   (e.g. `control` and `target` in TransversalCNOT) execute in separate
   time slots instead of running in-step, in-parallel in the same TICK.

The **memory experiment** (`CSSMemoryExperiment`) achieves full parallelism
via `_emit_interleaved_round()` — X and Z CX gates share the same TICK
layer. The gadget experiment bypasses interleaving entirely.

---

## 2. Root Cause Analysis

### 2.1 Stim Circuit Level: `_emit_parallel()` in `detector_emission.py`

`MemoryRoundEmitter._emit_parallel()` (line 826) is the entry point for
multi-block parallel extraction. Its current logic:

```python
# NON-ANCHOR path (standard case for all EC rounds except first/last):
for b in self.builders:
    b.emit_z_layer(circuit, ...)   # Builder 1 full Z, then Builder 2 full Z
for b in self.builders:
    b.emit_x_layer(circuit, ...)   # Builder 1 full X, then Builder 2 full X
```

Each `emit_z_layer()` call invokes `_emit_z_round()` on the builder, which
emits the complete Z measurement subcircuit:

```
R z_ancillas → TICK → CX_phase0 → TICK → CX_phase1 → ... → TICK → MR z_ancillas
```

**Problem**: Builder 1's Z-round fully completes (with its own TICKs)
before Builder 2's Z-round starts. This means:
- Block 1 Z CXs are in TICK epochs 1-5 (example)
- Block 2 Z CXs are in TICK epochs 6-10
- Block 1 X CXs are in TICK epochs 11-15
- Block 2 X CXs are in TICK epochs 16-20

All 4 groups occupy **different TICK regions**. There is zero parallelism
between blocks, and X/Z within a block are fully serial.

**Contrast with single-block memory experiment**:
`_emit_interleaved_round()` emits X and Z CX gates in the SAME TICK:

```
RX x_anc, R z_anc → TICK
  → CX(x_anc→data) + CX(data→z_anc) for phase 0   (SAME TICK)
  → TICK
  → CX(x_anc→data) + CX(data→z_anc) for phase 1   (SAME TICK)
  → ... → TICK → MRX x_anc, MR z_anc → TICK
```

This produces 4 CX TICK layers (for geometric 4-phase schedule) with
X+Z interleaved, halving the depth.

### 2.2 Metadata Level: `from_gadget_experiment()` in `pipeline.py`

`from_gadget_experiment()` builds `per_block_stabilizers[block_name]` with
a **combined** CNOT schedule:

```python
combined_layers: List[List[Tuple[int, int]]] = []
if x_stab.cnot_schedule:
    combined_layers.extend(x_stab.cnot_schedule)   # 4 X layers
if z_stab.cnot_schedule:
    combined_layers.extend(z_stab.cnot_schedule)   # 4 Z layers
# → 8 total layers instead of 4 interleaved
```

This produces **8 sequential CNOT layers** per block per EC round (4 X then
4 Z) instead of 4 interleaved layers (each containing X+Z pairs).

### 2.3 Routing Level: `derive_ms_pairs_from_metadata()` in `gadget_routing.py`

`derive_ms_pairs_from_metadata()` reads `per_block_stabilizers[block_name].cnot_schedule`
and produces one MS round per CNOT layer. With 8 layers, this creates
8 routing rounds per block per EC phase. Each routing round requires a
separate reconfig → drain → MS execution cycle.

With proper interleaving (4 layers), this would be halved.

### 2.4 Inter-block routing: `decompose_into_phases()` with merge

`decompose_into_phases()` iterates over active blocks per EC phase and
**merges MS pairs by round index** (line 900):
```python
for r_idx, round_pairs in enumerate(block_ms):
    while len(ms_pairs_all_rounds) <= r_idx:
        ms_pairs_all_rounds.append([])
    ms_pairs_all_rounds[r_idx].extend(round_pairs)
```

This means if both blocks have 8 rounds, they are merged at matching
indices: round 0 of block A merges with round 0 of block B, etc.
If the CNOT schedules are aligned (same geometric code), the merged
rounds contain corresponding X or Z pairs from both blocks.

**But**: because metadata has 8 layers (X||Z not X+Z), the merging
produces 8 merged rounds where each round contains only X-type or
only Z-type pairs. With interleaving, it would produce 4 merged rounds,
each containing both X and Z pairs from both blocks.

### 2.5 Compiler-path (`ionRoutingGadgetArch`) via `toMoveOps`/`parallelPairs`

When routing goes through the full compiler path, MS pairs come from
`toMoveOps` (extracted from the stim circuit's CX gates grouped by TICK).
Because `_emit_parallel()` puts each block's Z and X rounds in separate
TICKs, `parallelPairs` has one entry per TICK-delimited CX group:
- group 0: block1 Z phase0 pairs
- group 1: block1 Z phase1 pairs
- ...
- group 4: block2 Z phase0 pairs
- ...
- group 8: block1 X phase0 pairs
- ...etc

This produces 16 separate MS rounds per EC round (4 Z × 2 blocks + 4 X × 2 blocks)
instead of 4 merged rounds (each with X+Z from both blocks).

### 2.6 `cx_per_ec_round` Metadata

`cx_per_ec_round` is computed as:
```python
_cx += len(x_info.cnot_schedule)   # 4
_cx += len(z_info.cnot_schedule)   # 4
_cx_per_ec = 8
```

With interleaving, this should be 4 (the number of geometric phases).
This value is used by `_build_plans_from_compiler_pairs` to determine
how many `parallelPairs` entries belong to each EC phase.

---

## 3. Fix Strategy

### Fix A: Interleave the stim circuit emission (`_emit_parallel`)

**File**: `src/qectostim/experiments/detector_emission.py`

**Change**: Replace the sequential per-builder Z-then-X emission with
a **cross-builder interleaved** emission that:
1. Collects all builders that support interleaving
2. For each geometric phase `i` (0..3 for 4-phase schedule):
   - Emit all builders' X+Z CX gates for phase `i` in the SAME TICK
3. Handles reset (before CX phases) and measurement (after) across all builders

**New method**: `_emit_parallel_interleaved()` on `MemoryRoundEmitter`

**Algorithm**:
```
# Phase 1: Reset all ancillas for all builders
for b in builders:
    emit RX on b.x_ancillas (normal) or R (swapped) 
    emit R on b.z_ancillas (normal) or RX (swapped)
emit TICK

# Phase 2: Interleaved CX phases
n_phases = max(len(b._x_schedule) for b in builders if b._can_interleave())
for phase_idx in 0..n_phases-1:
    if phase_idx > 0: emit TICK
    for b in builders:
        emit b's X CX for phase_idx
        emit b's Z CX for phase_idx

# Phase 3: Measure all ancillas for all builders
emit TICK
for b in builders:
    emit MRX/MR on b.x_ancillas + MR/MRX on b.z_ancillas
    emit detectors for b

emit TICK   # final separator
```

**For anchor rounds**: Use the same structure but respect per-builder
`first_basis` ordering (which affects reset/measure basis, not CX ordering).

**Fallback**: If any builder cannot interleave (`_can_interleave()` is
False), fall back to the current sequential emission.

### Fix B: Interleave the CNOT schedule metadata (`from_gadget_experiment`)

**File**: `src/qectostim/experiments/hardware_simulation/core/pipeline.py`

**Change**: In `from_gadget_experiment()`, interleave X and Z CNOT layers
phase-by-phase instead of concatenating:

```python
# BEFORE (broken):
combined_layers = x_layers + z_layers   # [X0,X1,X2,X3,Z0,Z1,Z2,Z3]

# AFTER (fixed):
combined_layers = interleave(x_layers, z_layers)  # [X0+Z0, X1+Z1, X2+Z2, X3+Z3]
```

Each combined layer contains both X-type and Z-type CX pairs for the
same geometric phase, producing 4 layers instead of 8.

**Also update** `cx_per_ec_round` to reflect 4 phases (not 8).

### Fix C: Update `ms_pair_count` in PhaseInfo

**File**: `src/qectostim/experiments/hardware_simulation/core/pipeline.py`

The `ms_pair_count` for each EC phase is computed as:
```python
_ec_pre_ms = (_cx * rounds_before)  # e.g. 8 * 2 = 16
```

With interleaving, `_cx` becomes 4 (geometric phases), so:
```python
_ec_pre_ms = (4 * 2) = 8
```

But the pairs per round are now larger (each round has X+Z combined),
so the total pair count stays the same — just distributed across
fewer, larger rounds. The `ms_pair_count` needs to reflect the total
number of `parallelPairs` entries consumed, which is now
`n_interleaved_layers * num_rounds` instead of `(n_x_layers + n_z_layers) * num_rounds`.

**Actually**: `ms_pair_count` counts the number of `parallelPairs` entries
(i.e., the number of CX TICK layers), not the number of ion pairs. With
interleaving of the stim circuit, the number of TICK layers per EC round
is halved. So if the memory path uses `cx_per_ec_round = 4` and the stim
circuit also has 4 CX TICKs per round, both paths agree.

### Fix D: Ensure `_build_round_signature` still works

The round signature is built from X and Z stabilizer info. Interleaving
the CNOT schedule in the metadata changes the signature. But since
`_build_round_signature` runs on per-block X/Z StabilizerInfo separately
(not the combined schedule), it should not be affected.

---

## 4. Detailed Code Changes

### 4.1 `detector_emission.py` — `_emit_parallel()`

Replace `_emit_parallel()` body to use interleaved emission when all
builders support it.

Key cases:
1. **Non-anchor round**: Interleave all builders' X+Z CX in shared TICKs
2. **Anchor round**: Same interleaving, but with basis-dependent reset,
   anchor detector emission per builder
3. **Fallback**: If any builder can't interleave, fall back to current logic

The builder's `_emit_z_round`/`_emit_x_round` methods encapsulate reset
→ CX → measure internally, so we CANNOT use them for cross-builder
interleaving. Instead, we need to decompose the round structure:

- Reset: directly emit `RX`/`R` on each builder's ancillas
- CX: directly call `_emit_geometric_cnots` or use schedule info
- Measure: directly emit `MRX`/`MR` on each builder's ancillas
- Detectors: call `_emit_detectors_for_type` per builder

This means `_emit_parallel` needs access to builder internals. The
cleanest approach is to add a new method `emit_interleaved_cx_phase()`
on the builder that emits just the CX gates for one geometric phase,
without the surrounding reset/TICK/measure structure.

**New builder method**: `CSSStabilizerRoundBuilder.get_interleave_schedule()`
Returns the X and Z geometric schedules (list of (dx,dy) offsets) needed
for cross-builder interleaving. This avoids exposing all builder internals.

**New builder method**: `CSSStabilizerRoundBuilder.emit_cx_for_phase(circuit, phase_idx)`
Emits X+Z CX gates for a single geometric phase (no TICK, no reset, no measure).

**New builder method**: `CSSStabilizerRoundBuilder.emit_ancilla_reset(circuit)`
Emits the ancilla resets (RX for X-anc, R for Z-anc) respecting swap state.

**New builder method**: `CSSStabilizerRoundBuilder.emit_ancilla_measure(circuit)`
Emits ancilla measurements + detectors, respecting swap state.

### 4.2 `pipeline.py` — `from_gadget_experiment()`

Change CNOT schedule combination:
```python
# BEFORE:
combined_layers = []
if x_stab.cnot_schedule:
    combined_layers.extend(x_stab.cnot_schedule)
if z_stab.cnot_schedule:
    combined_layers.extend(z_stab.cnot_schedule)

# AFTER:
combined_layers = _interleave_cnot_schedules(x_stab.cnot_schedule, z_stab.cnot_schedule)
```

Where `_interleave_cnot_schedules(x_layers, z_layers)` zips X and Z
layers and concatenates pairs within each phase:
```python
def _interleave_cnot_schedules(x_layers, z_layers):
    if not x_layers and not z_layers:
        return []
    x = x_layers or []
    z = z_layers or []
    n = max(len(x), len(z))
    combined = []
    for i in range(n):
        layer = []
        if i < len(x): layer.extend(x[i])
        if i < len(z): layer.extend(z[i])
        if layer:
            combined.append(layer)
    return combined
```

Also update `_cx_per_ec` calculation:
```python
# BEFORE: _cx = len(x_schedule) + len(z_schedule) = 8
# AFTER:  _cx = max(len(x_schedule), len(z_schedule)) = 4
_cx = max(
    len(x_info.cnot_schedule) if x_info.cnot_schedule else 0,
    len(z_info.cnot_schedule) if z_info.cnot_schedule else 0,
)
```

And update `ms_pair_count` accordingly.

### 4.3 `css.py` — New helper methods for cross-builder interleaving

Add to `CSSStabilizerRoundBuilder`:

```python
def emit_ancilla_reset(self, circuit):
    """Emit ancilla resets for both X and Z types (no TICK)."""
    
def emit_cx_for_phase(self, circuit, phase_idx):
    """Emit X+Z CX gates for one geometric phase (no TICK)."""
    
def emit_ancilla_measure_and_detectors(self, circuit, emit_detectors):
    """Emit ancilla measurements and detectors (no TICK)."""
    
def n_interleave_phases(self):
    """Number of geometric phases (for cross-builder sync)."""
```

---

## 5. Test Plan

### 5.1 Unit Tests — Stim Circuit Structure

**Test**: Verify that the gadget stim circuit now has the same TICK/CX
structure as the memory experiment for corresponding EC rounds.

```python
def test_gadget_ec_round_interleaved():
    """Gadget EC rounds should have interleaved X+Z CX like memory."""
    code = RotatedSurfaceCode(distance=2)
    gadget = TransversalCNOTGadget()
    ft = FaultTolerantGadgetExperiment(codes=[code], gadget=gadget, ...)
    circuit = ft.to_stim()
    
    # Count CX gates per TICK layer
    # Should see 4 CX TICK layers per EC round (not 8)
    # Each TICK should have both X-type and Z-type CX gates
```

### 5.2 Unit Tests — Metadata CNOT Schedule

**Test**: Verify `per_block_stabilizers` has interleaved schedule.

```python
def test_metadata_cnot_schedule_interleaved():
    """CNOT schedule should be interleaved (4 layers, not 8)."""
    meta = QECMetadata.from_gadget_experiment(...)
    for block_name, stab in meta.per_block_stabilizers.items():
        # Should have 4 layers (not 8)
        assert len(stab.cnot_schedule) == 4
        # Each layer should have both X-type and Z-type pairs
```

### 5.3 Integration Tests — Routing Pair Counts

**Test**: Verify routing sees 4 MS rounds per EC round, not 8.

```python
def test_routing_pairs_interleaved():
    """EC phases should have 4 MS rounds per repeat, not 8."""
    plans = decompose_into_phases(...)
    for plan in plans:
        if plan.phase_type == 'ec':
            # Each EC round should have 4 MS rounds (interleaved)
            n_per_round = len(plan.ms_pairs_per_round)
            assert n_per_round == 4 * plan.num_rounds
```

### 5.4 Regression Tests — Existing Memory Experiment

**Test**: Memory experiment circuit must not change (it already interleaves).

```python
def test_memory_circuit_unchanged():
    """Memory experiment must produce identical circuit after refactor."""
    code = RotatedSurfaceCode(distance=2)
    mem = CSSMemoryExperiment(code=code, rounds=2, noise_model=None, basis='z')
    circuit = mem.to_stim()
    # Compare against saved reference circuit
```

### 5.5 Regression Tests — Detector Error Model

**Test**: The noisy simulation must still produce a valid DEM.

```python
def test_gadget_dem_valid():
    """Noisy gadget circuit must produce decomposable DEM."""
    # ... build noisy circuit ...
    dem = noisy_circuit.detector_error_model(
        decompose_errors=True, approximate_disjoint_errors=True,
    )
    assert dem.num_detectors > 0
```

### 5.6 Animation Verification

**Test**: Run compile_gadget_for_animation and verify:
- No "ion not in trap" errors during animation
- Reconfig count matches allOps (no dropped reconfigs)
- Parallel steps reduced by ~50% vs current

### 5.7 What Could Break

1. **Detector formulas**: Interleaving changes measurement indices within
   a TICK layer. The `ctx.add_measurement()` must be called in the right
   order. X measurements must still precede Z measurements (or the detector
   formulas will reference wrong indices).
   
   **Mitigation**: Ensure measurement order within the interleaved emission
   matches the order expected by `_emit_detectors_for_type`. The current
   interleaved round in `_emit_interleaved_round` already handles this
   correctly — X measurements first, Z measurements second.

2. **Round signature mismatch**: If `_build_round_signature` depends on
   the combined CNOT schedule order, changing from concatenated to
   interleaved could change the signature hash. This would break EC
   phase caching (identical phases wouldn't be detected).
   
   **Mitigation**: `_build_round_signature` builds from per-type X/Z
   StabilizerInfo, not the combined schedule. Verify this.

3. **`ms_pair_count` / `cx_per_ec_round`**: These values are used by
   `_build_plans_from_compiler_pairs` to split `parallelPairs` into
   phases. If the stim circuit now has 4 CX TICKs per EC round instead
   of 8, `parallelPairs` has 4 entries per round. The metadata
   `ms_pair_count` must match.
   
   **Mitigation**: Update `_cx_per_ec` and `ms_pair_count` together.

4. **Anchor round emission**: Anchor rounds (first/last) have special
   basis ordering requirements. The interleaved emission must still
   respect `first_basis` and emit anchor detectors correctly.
   
   **Mitigation**: Test anchor rounds explicitly.

5. **Stabilizer-swapped state**: After Hadamard gates, X↔Z are swapped.
   The interleaved emission must check `_stabilizer_swapped` per builder.
   
   **Mitigation**: Each builder tracks its own swap state. The new
   `emit_cx_for_phase()` method reads `self._stabilizer_swapped`.

---

## 6. Implementation Order

1. **Add builder helper methods** (`css.py`): `emit_ancilla_reset()`,
   `emit_cx_for_phase()`, `emit_ancilla_measure_and_detectors()`,
   `n_interleave_phases()`

2. **Rewrite `_emit_parallel()`** (`detector_emission.py`): Use new
   builder methods for cross-builder interleaved emission

3. **Interleave CNOT schedule** (`pipeline.py`): Change combined_layers
   from concatenation to interleaving

4. **Update `cx_per_ec_round`** (`pipeline.py`): Use `max(len_x, len_z)`
   instead of `len_x + len_z`

5. **Update `ms_pair_count`** (`pipeline.py`): Recalculate based on
   interleaved layer count

6. **Run tests**: Unit tests, regression tests, animation verification
