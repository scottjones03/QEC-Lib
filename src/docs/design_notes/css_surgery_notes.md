# CSS Code Surgery Protocol

This document describes the fault-tolerant logical CNOT implementation via CSS code surgery,
based on the protocols from:
- **Cowtan & Burton (2024)**: "Generic Fault-Tolerant CNOT for CSS Codes"
- **Poirson et al. (2025)**: "Generalised Lattice Surgery"

## Overview

CSS code surgery enables a **universal logical CNOT** between ANY two logical qubits of
ANY CSS codes. Unlike transversal CNOT (which only works within certain code families),
surgery works for:
- Homogeneous pairs (surface code ↔ surface code)
- Heterogeneous pairs (color code ↔ surface code, toric code ↔ rotated surface, etc.)
- Different-distance codes

The key insight is that a logical CNOT can be decomposed into two joint measurements:
1. **ZZ measurement**: Couples Z operators between control and target
2. **XX measurement**: Couples X operators between control and target

With appropriate Pauli frame corrections based on measurement outcomes.

## Protocol Structure

### Phase 1: Pre-Surgery Stabilizer Rounds

Before surgery, both codes must be in valid codespaces. We run `num_rounds_before`
stabilizer measurement rounds on each code to:
1. Verify encoding
2. Establish detector baseline
3. Detect any pre-existing errors

```
For each code (control, target):
    For round in range(num_rounds_before):
        Measure X stabilizers → MR
        Measure Z stabilizers → MR
        Emit time-like DETECTOR comparing with previous round
```

### Phase 2: ZZ Merge

The ZZ merge couples the logical Z operators of control and target. This is done by:

1. **Identify boundary qubits**: Find qubits in `logical_z_support(control)` and
   `logical_z_support(target)` that can be coupled. For topological codes, these
   are typically boundary qubits; for general CSS codes, we use bridge ancillas.

2. **Prepare bridge ancillas**: Allocate `d` ancilla qubits (where `d` is the
   code distance) in the `|0⟩` state.

3. **Merge stabilizer**: Create a "merged" Z stabilizer spanning both codes:
   ```
   Z_merged = Z_boundary_control ⊗ Z_bridge ⊗ Z_boundary_target
   ```

4. **Measure merged stabilizer** for `num_merge_rounds`:
   - CNOT from control boundary qubits to bridge
   - CNOT from target boundary qubits to bridge
   - Measure bridge in Z basis
   - This projects onto the merged stabilizer eigenspace

5. **Emit detectors**: Compare merged measurements with pre-surgery Z stabilizers.

```
Bridge ancillas: qubits B_0, B_1, ..., B_{d-1}
Control boundary: qubits C_0, C_1, ..., C_{d-1}  
Target boundary: qubits T_0, T_1, ..., T_{d-1}

For each merge round:
    R(B_i)  // Reset bridge
    TICK
    CNOT(C_i, B_i) for all i  // Control → Bridge
    TICK
    CNOT(T_i, B_i) for all i  // Target → Bridge
    TICK
    MR(B_i)  // Measure bridge
    DETECTOR comparing with previous
```

### Phase 3: XX Merge

Similar to ZZ merge, but for logical X operators:

1. **Identify X boundaries**: Find qubits in `logical_x_support()`.

2. **Prepare bridge ancillas** in `|+⟩` state (or reuse with H gate).

3. **Merge X stabilizer**:
   ```
   X_merged = X_boundary_control ⊗ X_bridge ⊗ X_boundary_target
   ```

4. **Measure merged X stabilizer**:
   - H on bridge ancillas
   - CNOT from bridge to control boundary (reversed direction for X)
   - CNOT from bridge to target boundary
   - H on bridge
   - Measure bridge in Z basis

```
For each merge round:
    R(B_i)
    H(B_i)  // Prepare |+⟩
    TICK
    CNOT(B_i, C_i) for all i  // Bridge → Control
    TICK
    CNOT(B_i, T_i) for all i  // Bridge → Target
    TICK
    H(B_i)
    MR(B_i)
    DETECTOR
```

### Phase 4: Post-Surgery Stabilizer Rounds

After surgery, run `num_rounds_after` stabilizer rounds to:
1. Verify the codes are still in valid codespaces
2. Detect any surgery-induced errors
3. Re-establish baseline for future operations

```
For each code (control, target):
    For round in range(num_rounds_after):
        Measure X stabilizers
        Measure Z stabilizers
        Emit time-like DETECTOR
```

### Phase 5: Pauli Frame Update

The measurement outcomes determine Pauli corrections:
- ZZ measurement outcome `m_z`: If `m_z = 1`, apply Z correction to target
- XX measurement outcome `m_x`: If `m_x = 1`, apply X correction to target

In Stim, these corrections are tracked via the Pauli frame and feed-forward,
or the decoder handles them implicitly.

## Boundary Qubit Computation

The crucial step is identifying which qubits to use for the merge. For general
CSS codes, we use the `logical_x_support()` and `logical_z_support()` methods.

### For Topological Codes (Surface/Toric)

Boundaries are geometric:
- **Logical Z**: Runs along "rough" boundaries (Z-type)
- **Logical X**: Runs along "smooth" boundaries (X-type)

The boundary qubits are those where the logical operator has support along one edge.

### For Non-Topological CSS Codes

We compute boundaries algebraically:
1. Get `logical_z_support(control)` and `logical_z_support(target)`
2. Minimum overlap: The merge uses min(len(support_c), len(support_t)) ancillas
3. Pair qubits by position in the logical operator string

## Detector Continuity

Critical for decoder operation: detectors must maintain continuity across phases.

### Within Pre/Post Memory

Time-like detectors compare syndrome at time t with time t-1:
```
DETECTOR rec[-1] rec[-(n_stabs+1)]  // Current vs previous
```

### At Surgery Boundaries

When transitioning from pre-surgery to surgery:
- Last pre-surgery Z measurement for boundary stabilizers
- First merge Z measurement
- These should agree (up to the intentional merge)

Special handling:
1. Boundary stabilizers get "split" during surgery
2. Merged stabilizer replaces individual boundary stabilizers
3. After surgery, boundary stabilizers are "restored"

Detector strategy:
```
# Pre-surgery last round
prev_z_boundary = record_measurement(boundary_z_stab)

# Surgery: the merged stabilizer
merged_meas = record_measurement(bridge_ancilla)

# Post-surgery first round
new_z_boundary = record_measurement(boundary_z_stab)

# Detector: prev_z ⊕ merged ⊕ new_z should be deterministic
DETECTOR(prev_z_boundary, merged_meas, new_z_boundary)
```

## Observable Tracking

The logical CNOT transforms observables:
- **Control Z**: Unchanged
- **Control X**: X_control ⊗ X_target (becomes product)
- **Target Z**: Z_control ⊗ Z_target (becomes product)
- **Target X**: Unchanged

After surgery, the observable includes measurements from both codes:
```
OBSERVABLE_INCLUDE(control_z_measurements, target_z_measurements)
```

## Implementation Checklist

### CSS Code Requirements

1. [ ] `logical_x_support(logical_idx)` → List[int]
2. [ ] `logical_z_support(logical_idx)` → List[int]
3. [ ] `hx`, `hz` parity check matrices
4. [ ] `n`, `k` properties

### SurgeryCNOT Gadget

1. [ ] `compute_layout()`: Position codes with bridge ancilla space
2. [ ] `compute_boundaries()`: Use logical support methods
3. [ ] `build_schedule()`:
   - Pre-surgery stabilizer rounds
   - ZZ merge (proper direction for CSS)
   - XX merge (proper direction for CSS)
   - Post-surgery stabilizer rounds
4. [ ] `to_stim()`:
   - Use scheduler pipeline
   - Emit QUBIT_COORDS
   - Emit time-like DETECTOR for each round
   - Emit boundary DETECTOR at surgery transitions
   - Emit OBSERVABLE_INCLUDE for transformed logical

### Testing

1. [ ] Zero logical error rate at p=0
2. [ ] Correct number of detectors
3. [ ] Detectors are all-zeros for noiseless circuit
4. [ ] Logical observable correctly tracks through surgery
5. [ ] Works for heterogeneous code pairs

## References

1. Cowtan, A., & Burton, S. (2024). Generic Fault-Tolerant CNOT for CSS Codes.
   arXiv:2401.XXXXX

2. Poirson, A., et al. (2025). Generalised Lattice Surgery. arXiv:2501.XXXXX

3. Horsman, C., et al. (2012). Surface code quantum computing by lattice surgery.
   New Journal of Physics, 14(12), 123011.

4. Litinski, D. (2019). A Game of Surface Codes. Quantum, 3, 128.
