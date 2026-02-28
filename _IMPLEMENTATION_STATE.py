"""
CRITICAL IMPLEMENTATION STATE - CSS SURGERY ANIMATION FIXES
============================================================

## Changes Already Made (in source files)

### Fix 1: _execute_ms_gates scan approach (qccd_WISE_ion_route.py ~line 2138)
Changed from indexing toMoves[round_idx] to scanning ALL operationsLeft:
- Builds remaining = [frozenset(p) for p in solved_pairs]
- Scans operationsLeft for 2-qubit ops matching remaining pairs
- Pops matched pairs from remaining to prevent double-execution
- Also changed candidate_gates logging from toMoves[ms_round_idx] to sum(len(s.solved_pairs) for s in round_steps)

### Fix 2: active_blocks=[] fallback (gadget_routing.py ~line 968 and ~1543)
Changed from: active_blocks = phase.active_blocks or list(all_block_names)
Changed to:   active_blocks = phase.active_blocks if phase.active_blocks is not None else list(all_block_names)
This prevents [] from being treated as falsy → all blocks.

## REMAINING WORK: Interleave EC rounds into gadget phases

### Problem:
The stim circuit for CSS surgery interleaves gadget bridge CX with EC CX:
  bridge_R1 → EC_R1(4 CX steps) → bridge_R2 → EC_R2(4 CX steps) → ...

But get_phase_pairs() only returns bridge pairs. The routing plan is MISSING
the interleaved EC rounds. Those EC ops will never be executed.

Stim CX structure for CSS surgery d=2 (28 CX total):
  CX[0-1]:   ZZ bridge R1 (4+4 pairs, qubit 21)
  CX[2-5]:   EC R1 (6 pairs each, all 3 blocks' stabilizers)
  CX[6-7]:   ZZ bridge R2 (4+4 pairs)
  CX[8-11]:  EC R2 (6 pairs each)
  CX[12-13]: XX bridge R1 (4+4 pairs, qubits 23,24)
  CX[14-17]: EC R3 (6 pairs each)
  CX[18-19]: XX bridge R2 (4+4 pairs)
  CX[20-23]: EC R4 (6 pairs each)
  CX[24-27]: post-EC (4 pairs each, blocks 0+2 only)

### Solution:
In decompose_into_phases (gadget_routing.py ~line 1040 in the gadget branch),
after getting gadget ms_pairs from get_phase_pairs, interleave EC rounds:

For each gadget phase with num_rounds > 0 and ms_pairs from get_phase_pairs:
1. Derive EC pairs using derive_ms_pairs_from_metadata for ALL blocks in all_block_names
2. For each gadget round i:
   a. Add gadget_round_i pairs (bridge pairs)
   b. Add EC pairs (4 rounds of stabilizer pairs for all blocks)
3. Result: ms_pairs = [bridge_R1, ec1, ec2, ec3, ec4, bridge_R2, ec1, ec2, ec3, ec4]

The phase type should remain "gadget" but the ms_pairs_per_round should
include both bridge and EC pairs in interleaved order.

### Key functions:
- decompose_into_phases: gadget_routing.py:929
- derive_ms_pairs_from_metadata: already exists, derives EC stabilizer pairs per block
- get_phase_pairs: CSSSurgeryCNOTGadget method, returns bridge-only pairs
- route_full_experiment_as_steps: gadget_routing.py:1776, routes phase plans

### Test commands:
cd "/Users/scottjones_admin/Library/Mobile Documents/com~apple~CloudDocs/Mac files/Repos/QECToStim"
WISE_INPROCESS_LIMIT=999999999 PYTHONPATH=src my_venv/bin/python SCRIPT.py

### Key imports:
from qectostim.codes.surface.rotated_surface import RotatedSurfaceCode; RotatedSurfaceCode(distance=2)
from qectostim.gadgets.css_surgery_cnot import CSSSurgeryCNOTGadget
from qectostim.experiments.ft_gadget_experiment import FaultTolerantGadgetExperiment
FaultTolerantGadgetExperiment(codes=[code], gadget=gadget, noise_model=None, num_rounds_before=1, num_rounds_after=1)

### QEC metadata for CSS surgery d=2:
phase[0]: init, active_blocks=None
phase[1]: stabilizer_round_pre, active_blocks=[]  ← EMPTY (no pre-gadget EC)
phase[2]: gadget ZZ merge, active_blocks=all 3, num_rounds=2
phase[3]: gadget ZZ split, num_rounds=0
phase[4]: gadget XX merge, active_blocks=all 3, num_rounds=2
phase[5]: gadget XX split, num_rounds=0
phase[6]: gadget Anc MX, num_rounds=0
phase[7]: stabilizer_round_post, active_blocks=['block_0','block_2'], num_rounds=1
phase[8]: measure

### Bridge double-counting fix (DONE in prior session):
layout.py line 1059: alloc._total_qubits = self._next_global_idx - len(self.bridge_ancillas)
(was: alloc._total_qubits = self.total_qubits)
"""
