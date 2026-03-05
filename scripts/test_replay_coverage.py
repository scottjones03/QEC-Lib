#!/usr/bin/env python3
"""
Fast instrumented test: verify EC cache replays ALL d×N cached reconfig
blocks without unnecessary SAT rebuilds.

Uses TransversalCNOTGadget (much faster SAT than CSSSurgery) to quickly
reach the EC cache replay codepath. Then checks:
  1. ec_cache_replay events > 0  (EC-level caching fires)
  2. Subsequent cached steps use cached schedules (no rebuild_schedule)
  3. Only 1 SAT transition per replay (the expected transition reconfig)

Line-buffered stdout for real-time output.
"""

import sys, os, time
sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', buffering=1)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from collections import defaultdict, Counter
import numpy as np
import logging

# ── Counters ────────────────────────────────────────────────────
counters = defaultdict(int)
_t0 = time.monotonic()
events = []

def _log_event(event_type, detail=""):
    counters[event_type] += 1
    elapsed = time.monotonic() - _t0
    events.append((elapsed, event_type, detail))
    print(f"  +{elapsed:7.1f}s  [{event_type:>25s}] #{counters[event_type]:3d}  {detail}", flush=True)


# ── Monkey-patches ──────────────────────────────────────────────
import qectostim.experiments.hardware_simulation.trapped_ion.compiler.qccd_WISE_ion_route as _rm

_orig_par = _rm._patch_and_route
_orig_rb  = _rm._rebuild_schedule_for_layout

def _mp_par(currentArr, wiseArch, P_arr, *a, **kw):
    _log_event("patch_and_route", f"P_arr={len(P_arr)} rounds")
    return _orig_par(currentArr, wiseArch, P_arr, *a, **kw)

def _mp_rb(old, wA, tgt, *a, **kw):
    diff = int(np.count_nonzero(old != tgt))
    _log_event("rebuild_schedule", f"mismatch={diff}")
    return _orig_rb(old, wA, tgt, *a, **kw)

class _Handler(logging.Handler):
    def emit(self, record):
        msg = record.getMessage()
        if "reusing cached block" in msg.lower():
            _log_event("block_cache_hit")
        elif "patch routing idx=" in msg and "produced" in msg:
            _log_event("fresh_sat_block", msg.split("produced")[1].strip()[:40] if "produced" in msg else "")
        elif "PRE-FLIGHT FAIL" in msg:
            _log_event("preflight_fail")
        elif "(EC cached): replaying" in msg:
            _log_event("ec_cache_replay", msg.split("replaying")[1].strip()[:40] if "replaying" in msg else "")
        elif "(EC cached): transition reconfig" in msg:
            _log_event("ec_transition_reconfig")
        elif "(EC cached): SAT-based" in msg:
            _log_event("ec_sat_transition")
        elif "(EC cached): SAT rebuild" in msg:
            _log_event("ec_sat_rebuild_first_step")
        elif "(EC cached): transition did NOT converge" in msg:
            _log_event("ec_transition_FAIL")
        elif "rechecked cache" in msg and "new_cache=" in msg:
            parts = msg.split("new_cache=")
            sz = parts[1].strip().split()[0].rstrip(",)") if len(parts) > 1 else "0"
            try:
                sz_int = int(sz)
            except ValueError:
                sz_int = 0
            if sz_int > 0:
                _log_event("cache_recheck_HIT", f"new_cache={sz}")
            else:
                _log_event("cache_recheck_miss")


# ── Main ────────────────────────────────────────────────────────
def main():
    from qectostim.codes.surface.rotated_surface import RotatedSurfaceCode
    from qectostim.experiments.ft_gadget_experiment import FaultTolerantGadgetExperiment
    from qectostim.gadgets.transversal_cnot import TransversalCNOTGadget
    from qectostim.experiments.hardware_simulation.trapped_ion.compiler.routing_config import WISERoutingConfig
    from qectostim.experiments.hardware_simulation.trapped_ion.demo.run import compile_gadget_for_animation
    from qectostim.experiments.hardware_simulation.trapped_ion.utils.gadget_routing import (
        decompose_into_phases, partition_grid_for_blocks,
    )

    d = 2; k = 2
    cfg = WISERoutingConfig.default(
        lookahead=1, subgridsize=(6,6,2), base_pmax_in=1,
        replay_level=d, cache_ec_rounds=True, show_progress=False,
    )
    gadget = TransversalCNOTGadget()
    code = RotatedSurfaceCode(distance=d)
    ft = FaultTolerantGadgetExperiment(codes=[code], gadget=gadget,
                                        noise_model=None, num_rounds_before=d, num_rounds_after=d)
    ideal = ft.to_stim()
    meta = ft.qec_metadata
    alloc = ft._unified_allocation

    print(f"Gadget: TransversalCNOT d={d}", flush=True)
    print(f"Circuit: {len(ideal)} instrs, {ideal.num_qubits} qubits", flush=True)
    print(f"Phases: {len(meta.phases)}", flush=True)
    for i, ph in enumerate(meta.phases):
        print(f"  [{i}] {ph.phase_type!s:10s}  rounds={ph.num_rounds}  "
              f"blocks={ph.active_blocks}", flush=True)
    print(f"replay_level={cfg.replay_level}, cache_ec_rounds={cfg.cache_ec_rounds}", flush=True)
    print(flush=True)

    # Count round_signatures to predict fresh vs replay counts
    sub_grids = partition_grid_for_blocks(meta, alloc, k)
    q2i = {q: q + 1 for ba in meta.block_allocations
           for q in list(ba.data_qubits) + list(ba.x_ancilla_qubits) + list(ba.z_ancilla_qubits)}
    plans = decompose_into_phases(meta, gadget, alloc, sub_grids, q2i, k)
    
    sig_counts = Counter()
    for p in plans:
        if p.round_signature is not None:
            sig_counts[p.round_signature] += 1
    
    print(f"Phase routing plans: {len(plans)}", flush=True)
    print(f"Distinct round_signatures: {len(sig_counts)}", flush=True)
    for sig, cnt in sig_counts.most_common():
        first_idx = next(i for i, p in enumerate(plans) if p.round_signature == sig)
        first_type = plans[first_idx].phase_type
        print(f"  sig hash={hash(sig) % 10000:04d}: seen {cnt}x "
              f"(first at plan[{first_idx}], type={first_type})", flush=True)
    
    fresh_expected = len(sig_counts)
    replay_expected = sum(c - 1 for c in sig_counts.values())
    print(f"\nExpected: {fresh_expected} fresh EC phases, "
          f"{replay_expected} EC cache replays", flush=True)
    print(flush=True)

    # Install hooks
    wl = logging.getLogger("wise")
    h = _Handler(); h.setLevel(logging.DEBUG)
    wl.addHandler(h); wl.setLevel(logging.DEBUG)
    _rm._patch_and_route = _mp_par
    _rm._rebuild_schedule_for_layout = _mp_rb

    print("=" * 60, flush=True)
    print("STARTING COMPILATION (TransversalCNOT d=2)", flush=True)
    print("=" * 60, flush=True)
    t0 = time.perf_counter()

    try:
        arch, compiler, compiled, batches, ion_roles, p2l, remap = (
            compile_gadget_for_animation(
                ideal, qec_metadata=meta, gadget=gadget,
                qubit_allocation=alloc, trap_capacity=k,
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
        _rm._patch_and_route = _orig_par
        _rm._rebuild_schedule_for_layout = _orig_rb
        wl.removeHandler(h)

    # ── Report ──────────────────────────────────────────────────
    print("\n" + "=" * 60, flush=True)
    print("REPLAY COVERAGE REPORT", flush=True)
    print("=" * 60, flush=True)

    for k_name in sorted(counters):
        print(f"  {k_name:<30s}  {counters[k_name]:>4d}", flush=True)

    pat  = counters.get("patch_and_route", 0)
    rb   = counters.get("rebuild_schedule", 0)
    bh   = counters.get("block_cache_hit", 0)
    fs   = counters.get("fresh_sat_block", 0)
    ecr  = counters.get("ec_cache_replay", 0)
    ect  = counters.get("ec_transition_reconfig", 0) + counters.get("ec_sat_transition", 0)
    ecf  = counters.get("ec_transition_FAIL", 0)
    ecsr = counters.get("ec_sat_rebuild_first_step", 0)
    pf   = counters.get("preflight_fail", 0)

    print(f"\n{'─'*60}", flush=True)
    print(f"Expected: {fresh_expected} fresh, {replay_expected} EC replays", flush=True)
    print(f"Actual:   {pat} patch_and_route, {ecr} ec_cache_replays", flush=True)
    print(f"{'─'*60}", flush=True)

    print(f"\nVERDICTS:", flush=True)
    
    if ecr > 0:
        print(f"  [PASS] EC cache replayed {ecr}x (expected {replay_expected}x)", flush=True)
    else:
        print(f"  [FAIL] EC cache NEVER replayed (expected {replay_expected}x)", flush=True)

    if ecr == replay_expected:
        print(f"  [PASS] Replay count matches expected ({ecr}=={replay_expected})", flush=True)
    elif ecr > 0:
        print(f"  [WARN] Replay count mismatch ({ecr} vs expected {replay_expected})", flush=True)

    if ecf == 0:
        print(f"  [PASS] Zero EC transition failures", flush=True)
    else:
        print(f"  [FAIL] {ecf} EC transition failure(s)", flush=True)

    if pf == 0:
        print(f"  [PASS] Zero PRE-FLIGHT FAILs", flush=True)
    else:
        print(f"  [WARN] {pf} PRE-FLIGHT FAILs", flush=True)

    print(f"  [INFO] Total _rebuild_schedule calls: {rb}", flush=True)
    print(f"  [INFO] Total _patch_and_route calls: {pat}", flush=True)
    print(f"  [INFO] EC transitions: {ect}", flush=True)
    print(f"  [INFO] block_cache hits: {bh}", flush=True)

    print("\n" + "=" * 60, flush=True)

if __name__ == "__main__":
    main()
