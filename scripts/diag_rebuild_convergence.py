#!/usr/bin/env python3
"""Diagnostic: trace _rebuild_schedule_for_layout convergence in detail.

Monkey-patches the function to log per-cycle, per-tiling, per-patch stats:
- How many BT pins were attempted vs assigned
- Mismatch before/after each cycle
- Which patches hit NoFeasibleLayoutError
- Whether merged schedule replay diverges
- How many cycles/tilings it takes (or doesn't converge)

Uses TransversalCNOT d=2 (fast SAT) with replay so we hit the transition
reconfig path that calls _rebuild_schedule_for_layout.
"""
import sys, os, time, logging
sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', buffering=1)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
from collections import defaultdict

# ── Patch _rebuild_schedule_for_layout to trace internals ──────
import qectostim.experiments.hardware_simulation.trapped_ion.compiler.qccd_WISE_ion_route as _rm

_orig_rebuild = _rm._rebuild_schedule_for_layout
_rebuild_call_id = 0

def _traced_rebuild(oldArr, wiseArch, target, *a, **kw):
    global _rebuild_call_id
    _rebuild_call_id += 1
    call_id = _rebuild_call_id
    
    n_rows, n_cols = oldArr.shape
    init_mismatch = int(np.count_nonzero(oldArr != target))
    allow_heuristic = kw.get('allow_heuristic_fallback', True)
    
    print(f"\n{'='*70}", flush=True)
    print(f"_rebuild_schedule_for_layout CALL #{call_id}", flush=True)
    print(f"  Grid: {n_rows}x{n_cols} = {n_rows*n_cols} cells", flush=True)
    print(f"  Initial mismatch: {init_mismatch}/{n_rows*n_cols} cells", flush=True)
    print(f"  allow_heuristic_fallback: {allow_heuristic}", flush=True)
    
    # Show which ions need to move
    ions_to_move = {}
    for r in range(n_rows):
        for c in range(n_cols):
            if oldArr[r,c] != target[r,c]:
                ion = int(oldArr[r,c])
                if ion != 0:
                    # Find where this ion needs to go
                    tgt_pos = None
                    for tr in range(n_rows):
                        for tc in range(n_cols):
                            if int(target[tr,tc]) == ion:
                                tgt_pos = (tr, tc)
                                break
                        if tgt_pos:
                            break
                    if tgt_pos:
                        dist = abs(r - tgt_pos[0]) + abs(c - tgt_pos[1])
                        ions_to_move[ion] = ((r,c), tgt_pos, dist)
    
    if ions_to_move:
        dists = [v[2] for v in ions_to_move.values()]
        print(f"  Ions needing movement: {len(ions_to_move)}", flush=True)
        print(f"  Manhattan distances: min={min(dists)}, max={max(dists)}, "
              f"mean={sum(dists)/len(dists):.1f}, total={sum(dists)}", flush=True)
        # Show distance distribution
        from collections import Counter
        dist_counts = Counter(dists)
        print(f"  Distance distribution: {dict(sorted(dist_counts.items()))}", flush=True)
        
        # Check for target column conflicts (the key constraint)
        from collections import defaultdict as dd
        col_demands = dd(list)  # target_col -> list of (ion, start_row)
        for ion, ((sr, sc), (tr, tc), d) in ions_to_move.items():
            col_demands[tc].append((ion, sr))
        
        conflicts = {tc: ions for tc, ions in col_demands.items() if len(ions) > 1}
        if conflicts:
            print(f"  Target column conflicts (multiple ions → same col):", flush=True)
            for tc, ions in sorted(conflicts.items()):
                source_rows = [sr for _, sr in ions]
                unique_rows = len(set(source_rows))
                print(f"    col {tc}: {len(ions)} ions from rows {source_rows} "
                      f"({unique_rows} unique rows → can pin {min(unique_rows, len(ions))}/cycle)", flush=True)
        
        # Check (start_row, target_col) key uniqueness
        key_demands = dd(list)
        for ion, ((sr, sc), (tr, tc), d) in ions_to_move.items():
            key_demands[(sr, tc)].append(ion)
        key_conflicts = {k: v for k, v in key_demands.items() if len(v) > 1}
        if key_conflicts:
            print(f"  (start_row, target_col) key conflicts: {len(key_conflicts)}", flush=True)
            for k, ions in list(sorted(key_conflicts.items()))[:5]:
                print(f"    key {k}: ions {ions}", flush=True)
            if len(key_conflicts) > 5:
                print(f"    ... {len(key_conflicts)-5} more", flush=True)
    
    t0 = time.perf_counter()
    result = _orig_rebuild(oldArr, wiseArch, target, *a, **kw)
    elapsed = time.perf_counter() - t0
    
    # Analyze result
    if result:
        final_layout = result[-1][0]
        final_mismatch = int(np.count_nonzero(final_layout != target))
        has_heuristic = any(s[1] is None for s in result)
        n_steps = len(result)
        
        print(f"\n  RESULT: {n_steps} steps in {elapsed:.2f}s", flush=True)
        print(f"  Final mismatch: {final_mismatch}/{n_rows*n_cols} cells", flush=True)
        print(f"  Converged: {final_mismatch == 0}", flush=True)
        print(f"  Has heuristic fallback: {has_heuristic}", flush=True)
        
        for i, (layout, sched, _) in enumerate(result):
            mm = int(np.count_nonzero(layout != target))
            sched_type = "SAT" if sched is not None else "HEURISTIC"
            sched_len = len(sched) if sched else 0
            print(f"    step {i}: mismatch={mm}, schedule={sched_type} ({sched_len} ops)", flush=True)
    else:
        print(f"\n  RESULT: EMPTY (0 steps) in {elapsed:.2f}s", flush=True)
    
    print(f"{'='*70}\n", flush=True)
    return result

_rm._rebuild_schedule_for_layout = _traced_rebuild


# ── Also trace _optimal_QMR_for_WISE to see BT pin success rates ──
from qectostim.experiments.hardware_simulation.trapped_ion.utils.qccd_operations import (
    GlobalReconfigurations,
)

_orig_qmr = GlobalReconfigurations._optimal_QMR_for_WISE

_qmr_call_id = 0
def _traced_qmr(*args, **kw):
    """Trace wrapper that forwards all args/kwargs unchanged."""
    global _qmr_call_id
    _qmr_call_id += 1
    
    # Extract what we need for logging from kwargs (callers use keyword args)
    BT = kw.get('BT', None) 
    P_arr = args[1] if len(args) > 1 else kw.get('P_arr', [[]])
    grid_origin = kw.get('grid_origin', (0,0))
    bt_soft = kw.get('bt_soft', False)
    patch_grid = args[0] if len(args) > 0 else None
    n_bt = sum(len(bt) for bt in BT) if BT else 0
    n_pairs = sum(len(p) for p in P_arr) if P_arr else 0
    
    # Only log for rebuild calls (P_arr=[[]] = no MS gates)
    is_rebuild = n_pairs == 0 and n_bt > 0
    
    if is_rebuild:
        shape = patch_grid.shape if patch_grid is not None else "?"
        print(f"  [QMR #{_qmr_call_id}] patch {shape} "
              f"origin={grid_origin} BT={n_bt} pins "
              f"bt_soft={bt_soft}", flush=True, end="")
    
    try:
        result = _orig_qmr(*args, **kw)
        if is_rebuild:
            # Check how many BT pins were achieved
            result_layout = result[0][0] if result[0] else None
            if result_layout is not None and BT:
                bt_achieved = 0
                bt_total = 0
                for bt_round in BT:
                    for ion_id, (tr, tc) in bt_round.items():
                        bt_total += 1
                        if int(result_layout[tr, tc]) == ion_id:
                            bt_achieved += 1
                print(f" → OK ({bt_achieved}/{bt_total} pins placed)", flush=True)
            else:
                print(f" → OK", flush=True)
        return result
    except Exception as e:
        if is_rebuild:
            print(f" → FAILED ({type(e).__name__})", flush=True)
        raise

GlobalReconfigurations._optimal_QMR_for_WISE = staticmethod(_traced_qmr)


# ── Logging ────────────────────────────────────────────────────
wl = logging.getLogger("wise")
wl.setLevel(logging.INFO)

class _RebuildHandler(logging.Handler):
    def emit(self, record):
        msg = record.getMessage()
        if "schedule-only rebuild" in msg:
            print(f"  [LOG] {msg}", flush=True)
        elif "PRE-FLIGHT" in msg:
            print(f"  [LOG] {msg[:120]}", flush=True)

h = _RebuildHandler()
h.setLevel(logging.INFO)
wl.addHandler(h)


# ── Run TransversalCNOT d=2 compilation ───────────────────────
def main():
    from qectostim.codes.surface.rotated_surface import RotatedSurfaceCode
    from qectostim.experiments.ft_gadget_experiment import FaultTolerantGadgetExperiment
    from qectostim.gadgets.transversal_cnot import TransversalCNOTGadget
    from qectostim.experiments.hardware_simulation.trapped_ion.compiler.routing_config import WISERoutingConfig
    from qectostim.experiments.hardware_simulation.trapped_ion.demo.run import compile_gadget_for_animation

    d = 2; k = 2
    cfg = WISERoutingConfig.default(
        lookahead=1, subgridsize=(6,6,2), base_pmax_in=1,
        replay_level=d, cache_ec_rounds=True, show_progress=False,
    )
    gadget = TransversalCNOTGadget()
    code = RotatedSurfaceCode(distance=d)
    ft = FaultTolerantGadgetExperiment(
        codes=[code], gadget=gadget, noise_model=None,
        num_rounds_before=d, num_rounds_after=d,
    )
    ideal = ft.to_stim()

    print(f"TransversalCNOT d={d}, replay_level={cfg.replay_level}, "
          f"cache_ec_rounds={cfg.cache_ec_rounds}", flush=True)
    print(f"Circuit: {len(ideal)} instrs, {ideal.num_qubits} qubits\n", flush=True)

    t0 = time.perf_counter()
    try:
        arch, compiler, compiled, batches, ion_roles, p2l, remap = (
            compile_gadget_for_animation(
                ideal, qec_metadata=ft.qec_metadata, gadget=gadget,
                qubit_allocation=ft._unified_allocation, trap_capacity=k,
                show_progress=False, max_inner_workers=4,
                routing_config=cfg,
            )
        )
        wall = time.perf_counter() - t0
        print(f"\nDONE in {wall:.1f}s", flush=True)
    except Exception as exc:
        wall = time.perf_counter() - t0
        print(f"\nFAILED after {wall:.1f}s: {exc}", flush=True)
        import traceback; traceback.print_exc()
    finally:
        # Restore originals
        _rm._rebuild_schedule_for_layout = _orig_rebuild
        GlobalReconfigurations._optimal_QMR_for_WISE = staticmethod(_orig_qmr)

if __name__ == "__main__":
    main()
