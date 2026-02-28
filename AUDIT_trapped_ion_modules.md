# Comprehensive Audit: `trapped_ion/` vs `old/src/` vs `trapped_ion_v2/`

## Executive Summary

| Module | trapped_ion/ Lines | old/ Equivalent | old/ Lines | In v2? | Port Complexity |
|---|---|---|---|---|---|
| compiler.py | 54 | — (re-export shim) | — | No | Trivial |
| compilers/base.py | 725 | — (new abstraction) | — | No | High |
| compilers/wise.py | 2,343 | best_effort_compilation_WISE.py | 672 | No | Very High |
| compilers/qccd.py | 290 | — (inline in qccd_circuit.py) | — | No | Medium |
| compilers/augmented_grid.py | 180 | — (inline in qccd_circuit.py) | — | No | Medium |
| compilers/networked.py | 109 | — | — | No | Low |
| execution.py | 572 | utils/execution.py | 472 | No | High |
| experiments.py | 840 | simulator/qccd_circuit.py | 1,037 | No | High |
| noise.py | 696 | utils/noise.py | 706 | No | High |
| simulator.py | 193 | simulator/qccd_circuit.py | 1,037 | No | Medium |
| visualization.py | 2,467 | utils/visualization.py | 728 | No | Medium |
| routing/ (8 files) | 6,951 | compiler/ (6 files) | 4,310 | No | Very High |
| viz/ (2 files) | 3,902 | — | — | No | Low |
| **TOTAL** | **19,322** | | **~7,925** | | |

### What exists in `trapped_ion_v2/` today
- `__init__.py` (113 lines) — re-exports
- `architecture.py` (1,260 lines) — nodes, graph, arch classes (ported from old/qccd_nodes + old/qccd_arch)
- `operations.py` (1,597 lines) — old physicalOperation()+run() closure pattern
- `physics.py` (35 lines) — re-exports from `trapped_ion.physics`
- **Total: 3,005 lines** — architecture + operations + physics only

---

## Module-by-Module Audit

---

### 1. `trapped_ion/compiler.py` (54 lines)

**Status:** Pure re-export shim — NOT broken itself, but imports from broken `trapped_ion.compilers`.

**Public API:**
- Re-exports: `PI`, `PI_2`, `PI_4`, `DecomposedGate`, `TrappedIonCompiler`, `WISECompiler`, `QCCDCompiler`, `AugmentedGridCompiler`, `NetworkedGridCompiler`

**Imports from trapped_ion:**
```python
from qectostim.experiments.hardware_simulation.trapped_ion.compilers import (...)
```

**Old equivalent:** None — the old code embeds compilation in `qccd_circuit.py` (inline `_parseCircuitString`, `processCircuit*` methods).

**Recommendation:** No action needed — this is just a re-export shim. Port the `compilers/` package and this works automatically.

**Estimated effort:** Trivial (just change import paths if needed)

---

### 2. `trapped_ion/compilers/` (6 files, 3,701 total lines)

#### 2a. `compilers/__init__.py` (54 lines)
Re-exports from base, wise, qccd, augmented_grid, networked.

#### 2b. `compilers/base.py` (725 lines)

**Status:** New abstraction layer — inherits from `core.compiler.HardwareCompiler`.

**Public API:**
- `DecomposedGate` — dataclass for decomposition sequences
- `TrappedIonCompiler(HardwareCompiler)` — ABC with:
  - `decompose_to_native(circuit) → NativeCircuit`
  - `map_qubits(circuit) → MappedCircuit`
  - `route(circuit) → RoutedCircuit`
  - `schedule(circuit) → ScheduledCircuit`
  - Gate decomposition tables (CNOT→MS+rotations, H→RY+RX, etc.)
- Constants: `PI`, `PI_2`, `PI_4`

**Key imports from trapped_ion:**
```python
from ...core.compiler import HardwareCompiler, CompilationPass, ...
from ...core.pipeline import NativeCircuit, MappedCircuit, ...
from ...core.gates import GateSpec, GateType
from ...core.operations import PhysicalOperation, GateOperation, ...
from ..physics import DEFAULT_CALIBRATION
TYPE_CHECKING: from ..architecture import TrappedIonArchitecture, WISEArchitecture
TYPE_CHECKING: from ..routing import WISERoutingConfig
```

**Old equivalent:** Gate decomposition logic is embedded in `old/src/simulator/qccd_circuit.py` lines 127-203 (`_parseCircuitString`), which decomposes CNOT→`[YRot, XRot, XRot, MS, YRot]` and CZ→`[YRot, XRot, YRot, XRot, MS, YRot, YRot, XRot]`. The decomposition formulas are the same.

**Is it functional?** Partially — gate decomposition tables are complete and correct. But it depends heavily on `core.compiler`, `core.pipeline`, `core.operations` (the "new framework" abstractions that may or may not be fully implemented).

**Recommendation:** HIGH complexity port. The decomposition logic itself is straightforward and matches the old code. The issue is the deep dependency on `core.*` abstractions. For v2, you'd need to either:
1. Port this with all `core.*` dependencies, or
2. Rewrite using the old-style inline approach (no ABC, no pipeline stages)

#### 2c. `compilers/wise.py` (2,343 lines)

**Status:** The most complex compiler — WISE grid with SAT routing.

**Public API:**
- `WISECompiler(TrappedIonCompiler)`:
  - `__init__(architecture, optimization_level, use_global_rotations, routing_config, use_junction_routing, partition_strategy)`
  - `decompose_to_native()` — gate decomposition
  - `map_qubits()` — spatial/gate-affinity qubit partitioning
  - `route()` — delegates to `ionRoutingWISEArch` or `ionRouting`
  - `schedule()` — wraps `paralleliseOperations`
  - `search_configs()` — parallel SAT config search (the old `search_configs_best_exec_time`)
  - `compile()` — full pipeline orchestrator

**Key imports:**
```python
from ...core.pipeline import NativeCircuit, MappedCircuit, RoutedCircuit, ...
from ...core.gates import GateSpec, GateType
from ...core.operations import GateOperation, TransportOperation, ...
from .base import TrappedIonCompiler, DecomposedGate
from ..physics import DEFAULT_CALIBRATION
```

**Old equivalent:** `old/src/compiler/best_effort_compilation_WISE.py` (672 lines) — contains `search_configs_best_exec_time()`.  The old pipeline calls `ionRoutingWISEArch` directly from `QCCDCircuit.processCircuitWiseArch()`.

**Is it functional?** The core routing logic delegates to `routing/qccd_WISE_ion_route.py` which is a port of the old working code. The WISE compiler wrapper adds `core.*` pipeline abstractions on top.

**Recommendation:** VERY HIGH complexity. The old `best_effort_compilation_WISE.py` + `processCircuitWiseArch` is the proven path. For v2, either:
1. Port the WISECompiler as-is with `core.*` deps, or
2. Create a thinner wrapper that calls `ionRoutingWISEArch` directly (like old code)

#### 2d. `compilers/qccd.py` (290 lines)

**Status:** General QCCD compiler for non-WISE architectures.

**Public API:**
- `QCCDCompiler(TrappedIonCompiler)`:
  - `decompose_to_native()` — reuses base decomposition
  - `map_qubits()` — sequential or cluster-based mapping
  - `route()` — delegates to greedy `ionRouting`
  - `schedule()` — delegates to `paralleliseOperations`
- `_restore_arrangement()` — helper to reset ion positions for animation replay

**Old equivalent:** Inline in `qccd_circuit.py`'s `processCircuit()` method.

**Recommendation:** MEDIUM complexity — mostly a thin wrapper.

#### 2e. `compilers/augmented_grid.py` (180 lines)

**Status:** Extends QCCDCompiler with Hungarian-matching cluster placement.

**Public API:**
- `AugmentedGridCompiler(QCCDCompiler)`:
  - Overrides `map_qubits()` with `regularPartition(isWISEArch=False)` + `hillClimbOnArrangeClusters`

**Key imports:**
```python
from ..architecture import AugmentedGridArchitecture, Ion
from ..routing import regularPartition, hillClimbOnArrangeClusters
```

**Old equivalent:** Inline in `qccd_circuit.py`'s `processCircuitAugmentedGrid()`.

**Recommendation:** MEDIUM complexity — self-contained, well-documented.

#### 2f. `compilers/networked.py` (109 lines)

**Status:** Extends QCCDCompiler with simple linear clustering.

**Public API:**
- `NetworkedGridCompiler(QCCDCompiler)`:
  - Overrides `map_qubits()` with `regularPartition` + `arrangeClusters`

**Old equivalent:** None (new architecture type).

**Recommendation:** LOW complexity — very small, well-contained.

**Special dependencies (compilers/):** `stim`, `numpy`, `scipy` (via routing imports)

---

### 3. `trapped_ion/execution.py` (572 lines)

**Status:** Creates `ExecutionPlan` from compilation metadata.

**Public API:**
- `TrappedIonExecutionPlanner`:
  - `plan_execution(circuit, compiled) → ExecutionPlan`
  - `_get_timing_from_compiled()` — extracts per-gate n̄, chain_length, mode snapshot
  - `_extract_chain_and_heating()` — per-batch motional quanta lookup
  - `_estimate_timing()` — fallback when no compilation data
  - `_get_fidelity()` — delegates to `IonChainFidelityModel`
  - `_get_swap_info()` — extracts gate-swap metadata from routed circuit
- `create_simple_execution_plan()` — convenience function

**Key imports from trapped_ion:**
```python
from ...core.execution import ExecutionPlan, OperationTiming, GateSwapInfo, IdleInterval
from ..physics import IonChainFidelityModel, DEFAULT_FIDELITY_MODEL, DEFAULT_CALIBRATION
TYPE_CHECKING: from ...core.pipeline import CompiledCircuit, ScheduledCircuit, ScheduledOperation
TYPE_CHECKING: from ..compiler import TrappedIonCompiler
TYPE_CHECKING: from ..noise import TrappedIonCalibration
```

**Old equivalent:** `old/src/utils/execution.py` (472 lines) — contains `OperationTiming`, `TransportInfo`, `IdleInterval`, `ExecutionPlan`, `build_execution_plan_from_operations()`. The old version works directly with `ParallelOperation` sequences; the new version works with `stim.Circuit` + `CompiledCircuit`.

**Is it functional?** Depends on `core.execution` dataclasses existing. The logic itself is sound.

**Recommendation:** HIGH complexity. This bridges between the Stim circuit instruction space and the physical operation space. The old version uses a different indexing scheme (operation-based vs instruction-based). For v2, you'd need to decide which indexing to use.

**Special dependencies:** `stim`

---

### 4. `trapped_ion/experiments.py` (840 lines)

**Status:** User-facing experiment class — the top-level entry point.

**Public API:**
- `TrappedIonExperiment(TrappedIonSimulator)`:
  - `__init__(code, architecture, gadget, rounds, compiler, hardware_noise, ...)`
  - `build_ideal_circuit() → stim.Circuit` (memory or FT gadget circuit)
  - `simulate(num_shots, decoder_name, gate_improvements) → HardwareSimulationResult`
  - `to_stim() → stim.Circuit` (with noise applied)
  - Properties: `qec_metadata`, `qubit_roles`, `data_qubit_indices`, `ancilla_qubit_indices`
  - `_build_memory_circuit()` — tries code.to_stim, CSSMemoryExperiment, StabilizerMemoryExperiment
  - `_build_gadget_circuit()` — delegates to FaultTolerantGadgetExperiment
  - `_get_execution_plan()` — compile + plan
  - `_compute_physical_xz_error_rates()` — analytical X/Z error accumulation (mirrors old `simulate()`)
- `TrappedIonGadgetExperiment` — placeholder (NotImplementedError)

**Key imports from trapped_ion:**
```python
from ..simulator import TrappedIonSimulator
from ...base import HardwareSimulationResult
from qectostim.codes.abstract_code import Code
from qectostim.codes.abstract_css import CSSCode
TYPE_CHECKING: from ..architecture import TrappedIonArchitecture
TYPE_CHECKING: from ..compiler import TrappedIonCompiler
TYPE_CHECKING: from ..noise import TrappedIonNoiseModel
```

**Old equivalent:** `old/src/simulator/qccd_circuit.py` (1,037 lines) — `QCCDCircuit.simulate()` is the equivalent entry point. The old version is a single monolithic class that does circuit parsing, ion mapping, routing dispatch, noise injection, and decoding all in one file.

**Is it functional?** The class structure is clean and well-documented, but depends on `TrappedIonSimulator`, the `core.*` pipeline, `qectostim.codes`, `qectostim.experiments.memory`, and `qectostim.experiments.ft_gadget_experiment` — many external dependencies.

**Recommendation:** HIGH complexity port. The `simulate()` method with gate-improvement sweeps + analytical X/Z error rates is particularly sophisticated. The old `QCCDCircuit.simulate()` does the same thing more simply. For v2, consider:
1. Port the full experiment class (needs many framework deps), or
2. Create a simpler wrapper that calls the old-style `simulate()` directly

**Special dependencies:** `stim`, `numpy`, `pymatching` (via old code)

---

### 5. `trapped_ion/noise.py` (696 lines)

**Status:** Timing-aware noise model for trapped ion hardware.

**Public API:**
- `TrappedIonCalibration(CalibrationData)` — calibration wrapper
- `TrappedIonNoiseModel(HardwareNoiseModel)`:
  - `apply_to_operation(operation) → List[NoiseChannel]` — basic gate noise
  - `apply_to_operation_timing(timing) → List[NoiseChannel]` — timing-aware with dynamic fidelity recomputation
  - `idle_noise(qubit, duration) → List[NoiseChannel]` — T2 dephasing
  - `swap_noise(swap_info) → List[NoiseChannel]` — SWAP = 3×MS gate noise
  - `apply_with_plan(circuit, plan) → stim.Circuit` — full noise injection with REPEAT preservation
  - `ms_gate_fidelity(chain_length, motional_quanta)` — delegates to IonChainFidelityModel
  - `single_qubit_gate_fidelity(chain_length, motional_quanta)`
  - `dephasing_fidelity(duration)`
  - `transport_error(operation_type)` — motional quanta per transport op
  - `mode_noise_callback` — hook for collaborator's correlated noise model

**Key imports from trapped_ion:**
```python
from qectostim.noise.hardware.base import HardwareNoiseModel, CalibrationData, NoiseChannel, NoiseChannelType
from ..physics import CalibrationConstants, DEFAULT_CALIBRATION, IonChainFidelityModel, DEFAULT_FIDELITY_MODEL
TYPE_CHECKING: from ...core.operations import PhysicalOperation
TYPE_CHECKING: from ...core.execution import ExecutionPlan, OperationTiming, GateSwapInfo
```

**Old equivalent:** `old/src/utils/noise.py` (706 lines) — nearly identical structure. Contains `NoiseChannel`, `TrappedIonCalibration`, `TrappedIonNoiseModel` with the same methods. The old version imports from `src.utils.physics` and `src.utils.execution` instead.

**Is it functional?** The noise model logic is complete and correct. Depends on `qectostim.noise.hardware.base` (framework ABC) and `core.execution` dataclasses.

**Recommendation:** HIGH complexity — but the old version is functionally identical. The key difference is the import paths. For v2:
1. Port the old `noise.py` directly (it imports from `src.utils.*` which are the old working modules)
2. Or adapt the new one to import from `trapped_ion_v2.physics` and `trapped_ion_v2.operations`

The `apply_with_plan()` method (lines 543-695) is critical — it implements the noise injection walk over the Stim circuit with REPEAT block preservation.

**Special dependencies:** `stim`, `numpy`

---

### 6. `trapped_ion/simulator.py` (193 lines)

**Status:** Base simulator class that integrates compiler + noise.

**Public API:**
- `TrappedIonSimulator(HardwareSimulator)`:
  - `__init__(code, architecture, compiler, hardware_noise, noise_model, metadata)`
  - `_create_default_compiler() → TrappedIonCompiler`
  - `apply_hardware_noise(circuit) → stim.Circuit` — creates ExecutionPlanner, builds plan, applies noise
  - `build_ideal_circuit()` — raises NotImplementedError (override in subclass)

**Key imports from trapped_ion:**
```python
from ...base import HardwareSimulator
from qectostim.codes.abstract_code import Code
TYPE_CHECKING: from ..architecture import TrappedIonArchitecture
TYPE_CHECKING: from ..compiler import TrappedIonCompiler
TYPE_CHECKING: from ..noise import TrappedIonNoiseModel
TYPE_CHECKING: from ...core.execution import ExecutionPlan
```

**Old equivalent:** Part of `old/src/simulator/qccd_circuit.py` — the `QCCDCircuit` class serves as both simulator and experiment.

**Is it functional?** Clean ABC that delegates to concrete implementations. Depends on `HardwareSimulator` base class from the framework.

**Recommendation:** MEDIUM complexity. This is a thin orchestration layer. The actual logic is in `execution.py` and `noise.py`.

**Special dependencies:** `stim`

---

### 7. `trapped_ion/visualization.py` (2,467 lines)

**Status:** Publication-quality matplotlib visualization for all architecture types.

**Public API:**
- `display_architecture(arch, ...) → (Figure, Axes)` — unified architecture renderer
- `animate_transport(arch, operations, ...) → FuncAnimation` — step-by-step ion transport animation
- `visualize_reconfiguration(arch, reconfig, ...)` — WISE reconfiguration phase visualization
- Helper dataclasses: `_TrapInfo`, `_JunctionInfo`, `_EdgeInfo`, `_GraphLayout`
- Layout extractors: `_extract_wise_layout()`, `_extract_qccd_layout()`
- Renderer: `_display_trapped_ion_graph()` — shared rendering pipeline
- Styling constants: `DPI`, `ION_RADIUS`, `SPACING`, color palettes, etc.

**Key imports from trapped_ion:**
```python
TYPE_CHECKING: from ..architecture import (
    TrappedIonArchitecture, QCCDArchitecture, WISEArchitecture,
    LinearChainArchitecture, QCCDGraph, QCCDNode, Ion,
    ManipulationTrap, Junction, Crossing,
)
```

**Old equivalent:** `old/src/utils/visualization.py` (728 lines) — simpler version with QCCD rendering only. The new version adds WISE, AugGrid, Networked, linear chain layouts, plus animation.

**Is it functional?** Yes — mostly self-contained. Uses `_mro_names()` instead of `isinstance()` to avoid stale module-cache issues. Only TYPE_CHECKING imports from `trapped_ion.architecture`.

**Recommendation:** MEDIUM complexity — largely self-contained. To port to v2:
1. Change TYPE_CHECKING imports to `trapped_ion_v2.architecture`
2. The `_mro_names()` dispatch means it works with any class that has the right name regardless of module origin

**Special dependencies:** `matplotlib`, `networkx` (optional)

---

### 8. `trapped_ion/routing/` (8 files, 6,951 total lines)

#### 8a. `routing/__init__.py` (108 lines)
Re-exports from all routing submodules.

**Key imports:**
```python
from ..architecture import Ion, QCCDNode, Trap, Junction, Crossing, ...
from ..operations import ParallelOperation
```

#### 8b. `routing/config.py` (145 lines)
`WISERoutingConfig` dataclass — SAT solver parameters.

**Old equivalent:** `old/src/utils/routing_config.py` (229 lines) — same class.

#### 8c. `routing/greedy_routing.py` (236 lines)
`ionRouting()` — greedy junction-based routing for non-WISE architectures.

**Key imports:**
```python
from ..architecture import Ion, QCCDNode, Trap, Junction, Crossing, ...
from ..operations import Operation, QubitGate, MSGate, GateSwap, CrystalRotation
```

**Old equivalent:** `old/src/compiler/qccd_ion_routing.py` (224 lines) — same algorithm, different import paths (`src.utils.qccd_nodes`, `src.utils.qccd_operations`).

#### 8d. `routing/process_supervisor.py` (181 lines)
`ProcessSupervisor` — tracks child PIDs for SAT solver cleanup.

**Old equivalent:** None (new infrastructure). Possibly extracted from `old/src/compiler/process_supervisor.py` (262 lines).

#### 8e. `routing/qccd_WISE_ion_route.py` (1,895 lines)
`ionRoutingWISEArch()` — WISE SAT-based routing with patch tiling.

**Key imports:**
```python
from ..architecture import Ion, QCCDNode, Trap, Junction, Crossing, QCCDComponent
from ..operations import Operation, QubitGate, MSGate, GateSwap, ParallelOperation
from .reconfiguration import ReconfigurationPlanner
from .sat_solver import NoFeasibleLayoutError
from .qccd_qubits_to_ions import *
```

**Old equivalent:** `old/src/compiler/qccd_WISE_ion_route.py` (1,709 lines) — very similar code with different imports (`src.utils.qccd_nodes`, `src.utils.qccd_operations`, `src.utils.qccd_arch`).

#### 8f. `routing/qccd_parallelisation.py` (390 lines)
`paralleliseOperations()`, `paralleliseOperationsSimple()`, `paralleliseOperationsWithBarriers()`, `calculateDephasingFromIdling()`, `calculateDephasingFidelity()`.

**Key imports:**
```python
from ..physics import DEFAULT_CALIBRATION
from ..architecture import Ion, Trap, Junction, ...
from ..operations import Operation, QubitGate, MSGate, GateSwap, ParallelOperation
```

**Old equivalent:** `old/src/compiler/qccd_parallelisation.py` (384 lines) — same algorithms.

#### 8g. `routing/qccd_qubits_to_ions.py` (557 lines)
`regularPartition()`, `arrangeClusters()`, `hillClimbOnArrangeClusters()` — BSP + Hungarian cluster placement.

**Key imports:**
```python
from ..architecture import Ion, Trap, Junction, ...
from ..operations import QubitGate, MSGate, GateSwap
```

**Old equivalent:** `old/src/compiler/qccd_qubits_to_ions.py` (528 lines) — same algorithms.

**Special dependencies:** `numpy`, `scipy` (linear_sum_assignment, distance_matrix)

#### 8h. `routing/reconfiguration.py` (1,775 lines)
`ReconfigurationPlanner` — global ion reconfiguration for WISE grid.

**Key imports:**
```python
from ..architecture import Ion, Trap, Junction, Crossing, QCCDWiseArch, ...
from ..physics import DEFAULT_CALIBRATION
```

**Old equivalent:** Part of `old/src/utils/qccd_operations.py` (3,914 lines) — the `GlobalReconfigurations` class.

#### 8i. `routing/sat_solver.py` (1,664 lines)
SAT/MaxSAT solver infrastructure — CNF builder, binary search, process-isolated solving.

**Key imports:**
```python
from pysat.card import CardEnc, EncType
from pysat.examples.rc2 import RC2
from pysat.formula import CNF, IDPool, WCNF
from pysat.solvers import Minisat22, Solver
from .process_supervisor import supervisor
```

**Old equivalent:** Part of `old/src/utils/qccd_operations.py` — the SAT solving was embedded in the operations module.

**Special dependencies:** `pysat` (python-sat), `numpy`

**Routing package summary:** VERY HIGH complexity overall. The routing files are direct ports of the old code with import paths changed from `src.utils.qccd_nodes`/`src.utils.qccd_operations` to `..architecture`/`..operations`. The algorithms are identical — the challenge is that they import from `trapped_ion.architecture` and `trapped_ion.operations`, which are the **new broken** versions. To work with v2, all these imports would need to change to `trapped_ion_v2.architecture` and `trapped_ion_v2.operations`.

---

### 9. `trapped_ion/viz/` (2 files, 3,902 lines)

#### 9a. `viz/__init__.py` (22 lines)
Re-exports from validate_schedule and visualization.

#### 9b. `viz/validate_schedule.py` (373 lines)
`ValidationResult` + `validate_schedule()` — checks trap capacity, MS co-location, ion conservation after routing.

**No trapped_ion imports** — works with plain strings and dicts. Self-contained.

**Old equivalent:** None — new infrastructure for debugging.

#### 9c. `viz/visualization.py` (3,529 lines)
Extended visualization with laser beams, Stim sidebar, detailed transport animation.

**Key imports:**
```python
TYPE_CHECKING: from ..architecture import (...)
```

**Old equivalent:** Extension of `old/src/utils/visualization.py`.

**Special dependencies:** `matplotlib`, `networkx` (optional)

**Recommendation:** LOW complexity for v2 — mostly self-contained. Just change TYPE_CHECKING imports.

---

## Old Code Cross-Reference

| Old File | Lines | What It Does | Ported To (trapped_ion/) |
|---|---|---|---|
| `utils/qccd_nodes.py` | 532 | Ion, Trap, Junction, Crossing, QCCDArch graph types | `architecture.py` + `trapped_ion_v2/architecture.py` |
| `utils/qccd_operations.py` | 3,914 | All operations (Split, Merge, Move, MS, GateSwap, Reconfig, SAT) | `operations.py` + `routing/reconfiguration.py` + `routing/sat_solver.py` |
| `utils/qccd_operations_on_qubits.py` | 380 | QubitOperation, TwoQubitMSGate, Measurement, etc. | `operations.py` (merged with above) |
| `utils/qccd_arch.py` | 441 | QCCDArch, QCCDWiseArch — graph builder | `architecture.py` |
| `utils/physics.py` | 823 | CalibrationConstants, IonChainFidelityModel | `physics.py` (shared, working) |
| `utils/noise.py` | 706 | TrappedIonNoiseModel, TrappedIonCalibration | `noise.py` |
| `utils/execution.py` | 472 | ExecutionPlan, OperationTiming, build_plan | `execution.py` |
| `utils/visualization.py` | 728 | Architecture display, transport animation | `visualization.py` + `viz/visualization.py` |
| `utils/routing_config.py` | 229 | WISERoutingConfig | `routing/config.py` |
| `utils/architecture.py` | 352 | TrappedIonArchitecture base | `architecture.py` |
| `compiler/qccd_ion_routing.py` | 224 | ionRouting (greedy) | `routing/greedy_routing.py` |
| `compiler/qccd_WISE_ion_route.py` | 1,709 | ionRoutingWISEArch (SAT) | `routing/qccd_WISE_ion_route.py` |
| `compiler/qccd_parallelisation.py` | 384 | paralleliseOperations | `routing/qccd_parallelisation.py` |
| `compiler/qccd_qubits_to_ions.py` | 528 | regularPartition, arrangeClusters | `routing/qccd_qubits_to_ions.py` |
| `compiler/best_effort_compilation_WISE.py` | 672 | search_configs_best_exec_time | `compilers/wise.py` |
| `compiler/_qccd_WISE_ion_routing.py` | 530 | SAT CNF builder, solver infra | `routing/sat_solver.py` |
| `simulator/qccd_circuit.py` | 1,037 | QCCDCircuit — simulate, decode, visualize | `experiments.py` + `simulator.py` |

---

## Recommendations for Porting to `trapped_ion_v2/`

### Priority Tiers

#### Tier 1 — Required for basic simulation (port FIRST)
1. **`noise.py`** — Port the old `utils/noise.py` directly, changing imports to `trapped_ion_v2`. This is the core noise model.
2. **`execution.py`** — Port the old `utils/execution.py`, adapting to use v2 operation types.
3. **`routing/` (all 8 files)** — These are already semi-ports of the old compiler code. Change all imports from `..architecture` → `trapped_ion_v2.architecture` and `..operations` → `trapped_ion_v2.operations`.

#### Tier 2 — Required for end-to-end experiments
4. **`compilers/base.py`** — The gate decomposition logic. Consider a simplified version that works without the `core.*` framework.
5. **`compilers/qccd.py`** + **`compilers/augmented_grid.py`** + **`compilers/networked.py`** — Thin wrappers over routing.
6. **`compilers/wise.py`** — The big one. Consider porting the old `best_effort_compilation_WISE.py` directly instead.
7. **`simulator.py`** — Thin orchestration layer.
8. **`experiments.py`** — User-facing entry point. Depends on everything above.

#### Tier 3 — Nice to have
9. **`visualization.py`** — Self-contained, change TYPE_CHECKING imports.
10. **`viz/`** — Self-contained utilities.
11. **`compiler.py`** — Re-export shim, trivial.

### Key Challenge: Import Path Rewiring

The central issue is that every module in `trapped_ion/` imports from:
- `trapped_ion.architecture` → needs `trapped_ion_v2.architecture`
- `trapped_ion.operations` → needs `trapped_ion_v2.operations`
- `trapped_ion.physics` → v2 already re-exports from `trapped_ion.physics` (OK)
- `core.compiler`, `core.pipeline`, `core.operations`, `core.execution` → NEW framework ABCs

The **routing/** files are closest to working because they import types (Ion, Trap, MSGate, etc.) that exist in both old and v2. The main work is a global find-and-replace of import paths.

The **compilers/** and **execution/noise/experiments** files have deeper dependencies on `core.*` framework abstractions (`HardwareCompiler`, `ExecutionPlan`, `HardwareNoiseModel`, `HardwareSimulator`, etc.) that may or may not be fully implemented.

### Recommended Strategy

**Option A (Fastest):** Port the old code directly into v2
- Copy `old/src/utils/noise.py` → `trapped_ion_v2/noise.py`, fix imports
- Copy `old/src/utils/execution.py` → `trapped_ion_v2/execution.py`, fix imports
- Copy `old/src/compiler/*.py` → `trapped_ion_v2/routing/`, fix imports
- Copy `old/src/simulator/qccd_circuit.py` → `trapped_ion_v2/simulator.py`, fix imports
- Avoid the `core.*` framework entirely

**Option B (Cleanest):** Port the new `trapped_ion/` code, rewiring imports
- All routing files: change `..architecture` → `trapped_ion_v2.architecture`, etc.
- Compilers: strip `core.*` dependencies or ensure they're implemented
- Noise/execution: use `core.execution` dataclasses or port the old standalone versions

**Estimated total effort:** 3-5 days for Option A, 1-2 weeks for Option B.
