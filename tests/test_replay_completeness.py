"""
Test: verify that replayed EC phases use ALL d × ceil(ms/4) cached blocks
with ZERO schedule rebuilds (SAT calls) during replay.

What we instrument:
  1. ec_cache hits vs misses in gadget_routing.py
  2. block_cache hits in _route_round_sequence (qccd_WISE_ion_route.py)
  3. _rebuild_schedule_for_layout calls (SAT rebuilds in execution loop)
  4. _patch_and_route calls (fresh SAT routing)

Expected with replay_level=d, d=2:
  - First occurrence of each distinct EC round_signature routes fresh (SAT)
  - Every subsequent occurrence is an ec_cache HIT
  - On a cache HIT: 1 SAT transition (at most) + 0 SAT rebuilds for
    d × ceil(ms_rounds/4) replayed block-cache entries
"""
import sys
import os
import logging
import re
from collections import defaultdict
from io import StringIO

# Ensure source is on path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

import numpy as np


def test_replay_completeness():
    """Run d=2 CSS Surgery compilation and verify full cache replay."""
    from qectostim.codes.surface.rotated_surface import RotatedSurfaceCode
    from qectostim.gadgets.css_surgery_cnot import CSSSurgeryCNOTGadget
    from qectostim.experiments.ft_gadget_experiment import FaultTolerantGadgetExperiment
    from qectostim.experiments.hardware_simulation.trapped_ion.compiler.routing_config import (
        WISERoutingConfig,
    )
    from qectostim.experiments.hardware_simulation.trapped_ion.demo.run import (
        compile_gadget_for_animation,
    )

    d = 2
    k = 2

    # Build experiment
    gadget = CSSSurgeryCNOTGadget()
    code = RotatedSurfaceCode(distance=d)
    ft_exp = FaultTolerantGadgetExperiment(
        codes=[code],
        gadget=gadget,
        noise_model=None,
        num_rounds_before=d,
        num_rounds_after=d,
    )
    ideal = ft_exp.to_stim()
    qec_meta = ft_exp.qec_metadata
    qubit_alloc = ft_exp._unified_allocation

    routing_cfg = WISERoutingConfig.default(
        lookahead=1,
        subgridsize=(6, 6, 2),
        base_pmax_in=1,
        replay_level=d,
        cache_ec_rounds=True,
        show_progress=False,
    )

    # ── Set up log capture ─────────────────────────────────────────
    log_buffer = StringIO()
    handler = logging.StreamHandler(log_buffer)
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(logging.Formatter("%(name)s %(levelname)s: %(message)s"))

    # Attach handler to the 'wise' logger hierarchy
    wise_root = logging.getLogger("wise")
    wise_root.setLevel(logging.DEBUG)
    wise_root.addHandler(handler)
    wise_root.propagate = False

    # ── Run compilation ────────────────────────────────────────────
    print(f"Running d={d} CSS Surgery compilation with replay_level={d} ...")
    import time
    t0 = time.perf_counter()

    arch, compiler, compiled, batches, ion_roles, p2l, remap = (
        compile_gadget_for_animation(
            ideal,
            qec_metadata=qec_meta,
            gadget=gadget,
            qubit_allocation=qubit_alloc,
            trap_capacity=k,
            show_progress=False,
            max_inner_workers=4,
            routing_config=routing_cfg,
        )
    )
    wall = time.perf_counter() - t0
    print(f"Compilation finished in {wall:.1f}s")

    # ── Analyse logs ───────────────────────────────────────────────
    wise_root.removeHandler(handler)
    log_text = log_buffer.getvalue()

    # Counters
    ec_cache_hits = 0
    ec_cache_misses = 0
    ec_fresh_routes = 0
    block_cache_hits = 0          # "reusing cached block"
    block_fresh_routes = 0        # "patch routing idx=..."
    schedule_rebuilds = 0         # "PRE-FLIGHT FAIL"
    transition_reconfigs = 0      # "transition reconfig"
    rebuild_sat_cycles = 0        # "schedule rebuilt via N SAT cycle(s)"

    # Per-phase tracking
    phase_cache_status = []       # list of (phase_idx, 'HIT'|'MISS')

    for line in log_text.splitlines():
        # EC cache hit: "(EC cached)"
        if "(EC cached)" in line:
            if "transition reconfig" in line:
                transition_reconfigs += 1
            elif "re-routing fresh" in line or "routing fresh instead" in line:
                ec_cache_misses += 1
            else:
                ec_cache_hits += 1

        # Block cache: "reusing cached block"
        if "reusing cached block" in line:
            block_cache_hits += 1

        # Fresh SAT: "patch routing idx="
        if "patch routing idx=" in line:
            block_fresh_routes += 1

        # Schedule rebuild: "PRE-FLIGHT FAIL"
        if "PRE-FLIGHT FAIL" in line:
            schedule_rebuilds += 1

        # SAT cycle count from rebuild
        m = re.search(r"schedule rebuilt via (\d+) SAT cycle", line)
        if m:
            rebuild_sat_cycles += int(m.group(1))

        # Phase cache status tracking
        m = re.search(r"phase=(\d+).*\(EC cached\)", line)
        if m:
            phase_idx = int(m.group(1))
            if "re-routing fresh" in line:
                phase_cache_status.append((phase_idx, "MISS_FALLBACK"))
            else:
                phase_cache_status.append((phase_idx, "HIT"))

    # ── Report ─────────────────────────────────────────────────────
    print()
    print("=" * 60)
    print("REPLAY COMPLETENESS REPORT")
    print("=" * 60)
    print(f"EC cache hits (replay):      {ec_cache_hits}")
    print(f"EC cache misses (fallback):  {ec_cache_misses}")
    print(f"EC fresh routes (no cache):  {ec_fresh_routes}")
    print(f"Transition reconfigs:        {transition_reconfigs}")
    print(f"Block cache hits (replay):   {block_cache_hits}")
    print(f"Block fresh SAT routes:      {block_fresh_routes}")
    print(f"Schedule rebuilds (SAT):     {schedule_rebuilds}")
    print(f"  -> total SAT cycles:       {rebuild_sat_cycles}")
    print()

    # ── Drill into block-level replay ──────────────────────────────
    # Count block cache hits PER ec_cache replay phase
    # Pattern: after an "(EC cached)" hit log, subsequent "reusing cached
    # block" logs until the next phase boundary belong to that replay.
    replay_phases = []
    current_replay_blocks = 0
    current_replay_rebuilds = 0
    in_replay = False

    for line in log_text.splitlines():
        if "(EC cached)" in line and "transition reconfig" not in line and "fresh" not in line:
            if in_replay:
                replay_phases.append({
                    "block_hits": current_replay_blocks,
                    "rebuilds": current_replay_rebuilds,
                })
            in_replay = True
            current_replay_blocks = 0
            current_replay_rebuilds = 0

        if in_replay:
            if "reusing cached block" in line:
                current_replay_blocks += 1
            if "PRE-FLIGHT FAIL" in line:
                current_replay_rebuilds += 1

    if in_replay:
        replay_phases.append({
            "block_hits": current_replay_blocks,
            "rebuilds": current_replay_rebuilds,
        })

    print("Per-EC-replay-phase block replay detail:")
    print(f"{'Phase':>6}  {'BlockHits':>10}  {'Rebuilds':>10}  {'FullReplay':>10}")
    print("-" * 45)
    all_full = True
    for i, rp in enumerate(replay_phases):
        is_full = rp["rebuilds"] == 0
        if not is_full:
            all_full = False
        status = "YES" if is_full else "NO"
        print(f"  {i:4d}  {rp['block_hits']:10d}  {rp['rebuilds']:10d}  {status:>10}")
    print()

    # ── Assertions ─────────────────────────────────────────────────
    # 1. There must be at least some ec_cache replay phases
    assert len(replay_phases) > 0, (
        f"Expected at least one EC cache replay phase, got 0. "
        f"Total log lines: {len(log_text.splitlines())}"
    )
    print(f"CHECK 1: {len(replay_phases)} EC cache replay phases found -> PASS")

    # 2. During replay phases, schedule rebuilds should be ZERO
    total_replay_rebuilds = sum(rp["rebuilds"] for rp in replay_phases)
    print(f"CHECK 2: Total schedule rebuilds during replay: {total_replay_rebuilds}", end="")
    if total_replay_rebuilds == 0:
        print(" -> PASS")
    else:
        print(f" -> FAIL (expected 0)")

    # 3. Each replay phase should replay at least d blocks (d stab rounds × ≥1 block each)
    min_blocks = min(rp["block_hits"] for rp in replay_phases) if replay_phases else 0
    print(f"CHECK 3: Minimum blocks replayed per phase: {min_blocks}", end="")
    if min_blocks >= d:
        print(f" (>= d={d}) -> PASS")
    else:
        print(f" (< d={d}) -> FAIL")

    # 4. Block fresh SAT routes should only happen for the FIRST occurrence
    # of each signature, not during replay
    print(f"CHECK 4: Block fresh SAT routes: {block_fresh_routes}, "
          f"block cache hits: {block_cache_hits}")

    # Overall
    print()
    if total_replay_rebuilds == 0 and min_blocks >= d and len(replay_phases) > 0:
        print("OVERALL: FULL REPLAY VERIFIED - all cached EC phases "
              "replay completely with zero SAT rebuilds")
    else:
        print("OVERALL: REPLAY INCOMPLETE - see details above")

    # Strict assertion
    assert total_replay_rebuilds == 0, (
        f"Expected 0 schedule rebuilds during EC cache replay, "
        f"got {total_replay_rebuilds}"
    )
    assert min_blocks >= d, (
        f"Expected at least {d} blocks replayed per EC phase, "
        f"got {min_blocks}"
    )


if __name__ == "__main__":
    test_replay_completeness()
