"""
Interpolated transport animation engine for old/ QCCD simulation.

Uses ``op.run()`` + ``arch.refreshGraph()`` as the authoritative mutation
engine (same as the old native notebook pattern), then reads ion positions
from arch internals via ``layout.read_ion_positions()`` to build pre-computed
snapshots.  Rendering uses the new-style custom matplotlib patches from
``display.py`` (FancyBboxPatch ions, laser beams, gate highlights) instead
of the old ``arch.display()`` / networkx delegation.

Features ported from trapped_ion/viz/visualization.py:
  - Smoothstep interpolation between snapshots
  - Laser beams on active ions during gate steps
  - Gate-hold frames for visibility
  - Progress bar with physical timing (us)
  - Stim sidebar (scrolling highlighted text)
  - Stim SVG timeslice panel
  - Transport arrows showing ion motion
  - Gate background tints
  - GridSpec multi-panel layout

IMPORTANT: ``_build_snapshots()`` is destructive — ``op.run()`` permanently
mutates the architecture.  Re-animation requires re-initialisation of the
``arch`` and ``operations``.
"""
from __future__ import annotations

import math
import warnings
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from .constants import (
    BG_COLOR,
    DPI,
    GATE_TIME_US,
    H_PASS_TIME_US,
    V_PASS_TIME_US,
    GATE_BG_TINTS,
    SIDEBAR_BG,
)
from .display import (
    _draw_ions_full,
    _draw_qccd_grid,
    _draw_wise_grid,
    _draw_transport_arrows,
    _extract_active_ions,
    _extract_per_ion_gate_kind,
    _extract_ms_pairs,
    _extract_ions_from_leaf,
    _gate_kind,
    _gate_kind_single,
    _is_transport,
    _mro_names,
    _require_matplotlib,
)
from .layout import read_ion_positions, read_trap_geometries
from .stim_panel import (
    draw_ops_sidebar,
    draw_progress_bar,
    draw_sidebar,
    draw_stim_svg,
    parse_stim_for_sidebar,
    parse_stim_timeslice_svg,
)


# =============================================================================
# Smoothstep easing
# =============================================================================

def _ease(t: float) -> float:
    """Hermite smoothstep: t*t*(3-2*t)."""
    t = max(0.0, min(1.0, t))
    return t * t * (3.0 - 2.0 * t)


# =============================================================================
# Pre-computation: snapshot building
# =============================================================================

def _read_trap_occupancy(arch: Any) -> Dict[int, int]:
    """Return {trap_idx: n_ions} for all manipulation traps."""
    occ: Dict[int, int] = {}
    for t in getattr(arch, "_manipulationTraps", []):
        occ[t.idx] = len(list(getattr(t, "ions", [])))
    return occ


def _build_snapshots(
    arch: Any,
    operations: Sequence[Any],
) -> Tuple[
    List[Dict[int, Tuple[float, float]]],
    List[Set[int]],
    List[Optional[str]],
    List[bool],
    List[str],
    List[float],
    List[Dict[int, int]],
]:
    """Run all operations sequentially, snapshot positions after each.

    Returns
    -------
    snapshots : list of {ion_idx: (x, y)}
        Position snapshot after each step (length = n_ops + 1).
    active_ions_per_step : list of set[int]
        Active ions at each step.
    gate_kind_per_step : list of str | None
        Gate kind per step ('ms', 'rotation', 'measure', 'reset', None).
    is_transport_per_step : list of bool
    labels_per_step : list of str
        Human-readable label for each step.
    durations_us : list of float
        Physical duration (us) for each step.
    trap_occupancy_per_step : list of {trap_idx: n_ions}
        Trap occupancy at each step for dynamic grid sizing.

    **DESTRUCTIVE**: ``op.run()`` permanently mutates ``arch``.
    """
    # Initial snapshot
    snapshots: List[Dict[int, Tuple[float, float]]] = [read_ion_positions(arch)]
    active_ions: List[Set[int]] = [set()]
    gate_kinds: List[Optional[str]] = [None]
    is_transport_list: List[bool] = [False]
    labels: List[str] = ["Initial"]
    durations: List[float] = [0.0]
    trap_occupancy: List[Dict[int, int]] = [_read_trap_occupancy(arch)]

    for i, op in enumerate(operations):
        # Run
        try:
            op.run()
            if hasattr(arch, "refreshGraph"):
                arch.refreshGraph()
        except Exception as exc:
            warnings.warn(f"animate_transport: step {i} failed: {exc}")
            # Snapshot whatever state we have
            if hasattr(arch, "refreshGraph"):
                try:
                    arch.refreshGraph()
                except Exception:
                    pass

        # Read the post-mutation positions
        snap = read_ion_positions(arch)
        snapshots.append(snap)

        # Active ions
        active_ions.append(_extract_active_ions(op))

        # Gate kind
        gk = _gate_kind(op)
        gate_kinds.append(gk)

        # Transport?
        is_transport_list.append(_is_transport(op))

        # Label
        lbl = getattr(op, "label", None) or type(op).__name__
        labels.append(lbl)

        # Duration — operationTime() returns seconds; convert to µs
        try:
            dur = float(getattr(op, "operationTime", lambda: 0)()) * 1e6
        except Exception:
            dur = 0.0
        if dur <= 0:
            if gk is not None:
                dur = GATE_TIME_US.get(gk, 10.0)
            else:
                dur = H_PASS_TIME_US
        durations.append(dur)

        # Trap occupancy for dynamic grid sizing
        trap_occupancy.append(_read_trap_occupancy(arch))

    return snapshots, active_ions, gate_kinds, is_transport_list, labels, durations, trap_occupancy


# =============================================================================
# Stim mapping helpers
# =============================================================================

def _build_stim_mapping(
    stim_circuit: Any,
    n_ops: int,
    gate_kinds: List[Optional[str]],
    active_ions_per_step: Optional[List[Set[int]]] = None,
    native_circuit: Any = None,
    ion_idx_remap: Optional[Dict[int, int]] = None,
    physical_to_logical: Optional[Dict[int, int]] = None,
    operations: Optional[Sequence[Any]] = None,
) -> Tuple[
    List["SidebarEntry"],
    Dict[int, Any],
    List[Optional[Set[int]]],
    List[Optional[int]],
    int,
]:
    """Build stim sidebar entries + per-step mapping.

    Uses per-sub-operation ``_stim_tag`` annotations (set during
    ``decompose_to_native``) to link each native instruction inside
    a ``ParallelOperation`` to the stim sidebar entry whose gate-type
    and qubit targets match.  Falls back to qubit-overlap heuristic
    when tags are absent, and finally to gate-count-proportional
    interpolation.

    Returns (sidebar_entries, tick_svg_cache, stim_lines_per_step,
             stim_tick_per_step, n_ticks).
    """
    sidebar = parse_stim_for_sidebar(stim_circuit)
    n_ticks = sum(1 for e in sidebar if e.kind == "tick")

    # Build tick_svg_cache
    tick_svg_cache: Dict[int, Any] = {}
    for tick_idx in range(max(1, n_ticks)):
        try:
            svg = parse_stim_timeslice_svg(stim_circuit, tick=tick_idx)
            if svg:
                tick_svg_cache[tick_idx] = svg
        except Exception:
            pass

    # TICK line locations in sidebar
    tick_line_idxs: List[int] = []
    for ei, entry in enumerate(sidebar):
        if entry.kind == "tick":
            tick_line_idxs.append(ei)

    # ---------------------------------------------------------------
    # Authoritative mapping via _stim_tag on sub-operations
    # ---------------------------------------------------------------
    stim_tick_per_step: List[Optional[int]] = [None] * (n_ops + 1)
    stim_lines_per_step: List[Optional[Set[int]]] = [None] * (n_ops + 1)

    _used_authoritative = False
    if active_ions_per_step is not None:
        _used_authoritative = _apply_authoritative_mapping(
            n_ops, gate_kinds, active_ions_per_step,
            ion_idx_remap, physical_to_logical,
            stim_tick_per_step, stim_lines_per_step,
            sidebar, tick_line_idxs, n_ticks,
            operations=operations,
        )

    # ---------------------------------------------------------------
    # Heuristic fallback: gate-count-proportional tick mapping
    # ---------------------------------------------------------------
    if not _used_authoritative:
        stim_tick_per_step[0] = 0
        cur_tick = 0
        gate_count_in_tick = 0
        # Count only actual gate steps (not transport) for threshold
        n_gate_ops = sum(1 for g in gate_kinds if g is not None)
        tick_gate_threshold = max(1, n_gate_ops // max(1, n_ticks) if n_ticks > 0 else 1)

        for step_idx in range(1, n_ops + 1):
            gk = gate_kinds[step_idx] if step_idx < len(gate_kinds) else None
            if gk is not None:
                gate_count_in_tick += 1
                if gate_count_in_tick >= tick_gate_threshold and cur_tick < n_ticks - 1:
                    cur_tick += 1
                    gate_count_in_tick = 0
            stim_tick_per_step[step_idx] = cur_tick

            if cur_tick < len(tick_line_idxs):
                tl = tick_line_idxs[cur_tick]
                next_tl = (tick_line_idxs[cur_tick + 1]
                           if cur_tick + 1 < len(tick_line_idxs)
                           else len(sidebar))
                stim_lines_per_step[step_idx] = set(range(tl, min(next_tl, tl + 15)))
            else:
                stim_lines_per_step[step_idx] = set()

    # Normalize viewBox across all ticks to prevent panel jumping.
    # Use the union bounding box padded to a square.  The content-tight
    # viewBox from parse_stim_timeslice_svg already clips to drawn
    # primitives, so the union here is much tighter than before.
    all_vbs = [svg["viewBox"] for svg in tick_svg_cache.values() if svg]
    if len(all_vbs) > 1:
        ux = min(v[0] for v in all_vbs)
        uy = min(v[1] for v in all_vbs)
        uw = max(v[0] + v[2] for v in all_vbs) - ux
        uh = max(v[1] + v[3] for v in all_vbs) - uy
        # Pad to square aspect ratio to prevent extreme squishing when
        # the layout is much wider than tall or vice versa (e.g. multi-
        # block gadgets with vertically stacked blocks).
        if uw > 0 and uh > 0:
            side = max(uw, uh)
            pad_x = (side - uw) / 2
            pad_y = (side - uh) / 2
            ux -= pad_x
            uy -= pad_y
            uw = side
            uh = side
        for svg in tick_svg_cache.values():
            if svg:
                svg["viewBox"] = (ux, uy, uw, uh)

    return sidebar, tick_svg_cache, stim_lines_per_step, stim_tick_per_step, n_ticks


def _apply_authoritative_mapping(
    n_ops: int,
    gate_kinds: List[Optional[str]],
    active_ions_per_step: List[Set[int]],
    ion_idx_remap: Optional[Dict[int, int]],
    physical_to_logical: Optional[Dict[int, int]],
    stim_tick_per_step: List[Optional[int]],
    stim_lines_per_step: List[Optional[Set[int]]],
    sidebar: list,
    tick_line_idxs: List[int],
    n_ticks: int,
    operations: Optional[Sequence[Any]] = None,
) -> bool:
    """Populate stim mapping arrays using per-sub-operation tag matching.

    Each ``QubitOperation`` produced by ``decompose_to_native()`` carries
    a ``_stim_tag = (category, frozenset_of_logical_qubits)`` attribute.
    For every ``ParallelOperation`` step the algorithm:

    1. Iterates over the native sub-operations inside the parallel step.
    2. Reads each sub-op's ``_stim_tag`` (category + logical qubits).
    3. Finds the **next unmatched** stim sidebar entry whose gate-type
       category matches and whose qubit targets overlap with the tag's
       logical qubits.  Consuming entries in order ensures that
       repeated stim blocks (e.g. multiple rounds of ``CX 4 1 3 6``)
       are matched sequentially rather than all at once.
    4. Highlights the matched sidebar entry indices for the step.

    Falls back to the prior qubit-overlap heuristic when ``_stim_tag``
    is absent (e.g. ``GateSwap`` ops inserted by routing).

    Returns ``True`` if the mapping was successfully applied.
    """
    if n_ticks == 0:
        return False

    # ------------------------------------------------------------------
    # 1. Build per-block entry sets (for interpolation fallback)
    # ------------------------------------------------------------------
    n_blocks = n_ticks + 1
    block_ranges: List[Tuple[int, int]] = []
    for bi in range(n_blocks):
        if bi == 0:
            start = 0
            end = tick_line_idxs[0] if tick_line_idxs else len(sidebar)
        else:
            start = tick_line_idxs[bi - 1]
            end = (tick_line_idxs[bi] if bi < len(tick_line_idxs)
                   else len(sidebar))
        block_ranges.append((start, end))

    block_gate_entries: List[Set[int]] = [set() for _ in range(n_blocks)]
    for bi, (s, e) in enumerate(block_ranges):
        for ei in range(s, e):
            if ei >= len(sidebar):
                break
            if sidebar[ei].kind == "gate":
                block_gate_entries[bi].add(ei)

    # ------------------------------------------------------------------
    # 2. Helper: sidebar-entry → tick block mapping
    # ------------------------------------------------------------------
    def _entry_block(ei: int) -> int:
        for bi in range(n_blocks):
            s, e = block_ranges[bi]
            if s <= ei < e:
                return bi
        return n_blocks - 1

    # Build set of sidebar indices that are gate entries (for validation)
    _gate_entry_indices: Set[int] = set()
    for ei, entry in enumerate(sidebar):
        if entry.kind == "gate":
            _gate_entry_indices.add(ei)

    # ------------------------------------------------------------------
    # 3. Direct provenance lookup
    # ------------------------------------------------------------------
    # Each native QubitOperation carries ``_stim_origin`` — the index of
    # the original (pre-decomposition) stim gate it came from.  This
    # index matches the sidebar entry order from parse_stim_for_sidebar()
    # exactly, so we just read it off.  No consumption pointers, no
    # category matching, no fallbacks needed.

    def _collect_sub_ops(op: Any) -> list:
        """Recursively get all leaf operations from a ParallelOperation."""
        cls = type(op).__name__
        if cls == "ParallelOperation":
            result = []
            for sub in getattr(op, "operations", []):
                result.extend(_collect_sub_ops(sub))
            return result
        return [op]

    anchors: List[Tuple[int, int]] = [(0, 0)]
    # Steps whose entries were resolved directly from _stim_origin
    # (ground-truth provenance).  Section 7 monotonicity enforcement
    # must not override these — they represent the real stim origin.
    _directly_matched: Set[int] = set()

    # Gate-kind → stim gate name category mapping for cross-validation.
    # Native rotations that are part of a CX/CZ decomposition must also
    # be allowed to highlight the parent 2Q stim gate, so "rotation"
    # includes all 2Q names as well.
    _2q_names = {"CX", "CZ", "CNOT", "XX", "ZZ", "SQRT_XX",
                 "XCX", "XCZ", "YCZ", "ISWAP", "SQRT_ZZ", "SPP"}
    _meas_names = {"M", "MR", "MX", "MY", "MZ", "MRX", "MRY", "MRZ", "MPP"}
    _reset_names = {"R", "RX", "RY", "RZ"}
    _rot_names = {"H", "S", "S_DAG", "X", "Y", "Z", "SQRT_X", "SQRT_Y",
                  "SQRT_Z", "SQRT_X_DAG", "SQRT_Y_DAG", "SQRT_Z_DAG",
                  "H_XY", "H_YZ", "C_XYZ", "C_ZYX"}
    _native_cat: Dict[str, Set[str]] = {
        "ms": _2q_names,
        "measure": _meas_names,
        # Resets may originate from combined stim ops like MR, MRX.
        "reset": _reset_names | _meas_names,
        # Rotations may originate from 2Q gate decompositions (e.g.
        # CX → RY,RX,RX,MS,RY) or basis-change rotations from
        # MRX/MX decompositions, so allow all categories.
        "rotation": _rot_names | _reset_names | _2q_names | _meas_names,
    }

    for si in range(1, n_ops + 1):
        gk = gate_kinds[si] if si < len(gate_kinds) else None
        if gk is None:
            continue

        matched_entries: Set[int] = set()

        if operations is not None and 0 < si <= len(operations):
            sub_ops = _collect_sub_ops(operations[si - 1])
            allowed_gates = _native_cat.get(gk, set())
            for sub in sub_ops:
                origin = getattr(sub, "_stim_origin", None)
                if origin is not None and origin >= 0 and origin in _gate_entry_indices:
                    # Cross-check: sidebar entry gate must match native op kind
                    entry = sidebar[origin]
                    entry_gate = (entry.gate or "").upper()
                    if not allowed_gates or entry_gate in allowed_gates:
                        matched_entries.add(origin)
                    # else: mismatch — skip this origin, fall through to heuristic

        if not matched_entries:
            # Fallback for un-tagged ops (e.g. transport steps):
            # use qubit-overlap heuristic
            ions = active_ions_per_step[si] if si < len(active_ions_per_step) else set()
            if ions:
                logical_qubits: Set[int] = set()
                for ion in ions:
                    disp = ion_idx_remap.get(ion, ion) if ion_idx_remap else ion
                    lq = physical_to_logical.get(disp, disp) if physical_to_logical else disp
                    logical_qubits.add(lq)

                # Map native gate kind → stim gate categories (reuse outer defs)
                allowed = _native_cat.get(gk, set())
                fz_lq = frozenset(logical_qubits)

                for ei, entry in enumerate(sidebar):
                    if entry.kind != "gate":
                        continue
                    g = (entry.gate or "").upper()
                    if g in allowed and frozenset(entry.qubits) & fz_lq:
                        matched_entries.add(ei)
                        break

        if matched_entries:
            stim_lines_per_step[si] = matched_entries
            blocks = {_entry_block(ei) for ei in matched_entries}
            earliest_block = min(blocks)
            stim_tick_per_step[si] = max(0, min(earliest_block, n_ticks - 1))
            anchors.append((si, earliest_block))
            # Mark as directly matched if entries came from _stim_origin
            if operations is not None and 0 < si <= len(operations):
                _directly_matched.add(si)

    # End anchor
    anchors.append((n_ops, n_blocks - 1))

    # Sort and deduplicate anchors
    seen_steps: Set[int] = set()
    unique_anchors: List[Tuple[int, int]] = []
    for a in sorted(anchors, key=lambda a: (a[0], a[1])):
        if a[0] not in seen_steps:
            seen_steps.add(a[0])
            unique_anchors.append(a)
    anchors = unique_anchors

    # ------------------------------------------------------------------
    # 6. Fill in non-gate steps by interpolation between anchors
    # ------------------------------------------------------------------
    for si in range(n_ops + 1):
        if stim_lines_per_step[si] is not None:
            continue

        prev_a = anchors[0]
        next_a = anchors[-1]
        for ai in range(len(anchors) - 1):
            if anchors[ai][0] <= si <= anchors[ai + 1][0]:
                prev_a = anchors[ai]
                next_a = anchors[ai + 1]
                break

        span = next_a[0] - prev_a[0]
        if span > 0:
            frac = (si - prev_a[0]) / span
            block = round(prev_a[1] + frac * (next_a[1] - prev_a[1]))
        else:
            block = prev_a[1]

        tick = max(0, min(block, n_ticks - 1))
        stim_tick_per_step[si] = tick

        clamped_block = max(0, min(block, n_blocks - 1))
        stim_lines_per_step[si] = set(block_gate_entries[clamped_block])

    # ------------------------------------------------------------------
    # 7. Monotonicity enforcement
    # ------------------------------------------------------------------
    # After scheduling reorders operations, native rotations that are
    # part of a CX decomposition may carry _stim_origin pointing to an
    # earlier stim line than the previously highlighted one.  This
    # causes the animation timeline to jump backwards, confusing the
    # viewer.  Enforce that tick and highlighted entries never go
    # backwards — once we advance to stim line N, we never go below N.
    #
    # EXCEPTION: Steps that were *directly matched* via _stim_origin
    # (ground-truth provenance) are exempt from entry filtering.
    # These represent the real stim instruction that generated the
    # native op.  Only interpolated/heuristic steps get filtered.
    # Tick monotonicity is still enforced universally for smooth
    # SVG panel progression.
    _high_water_tick = 0
    _high_water_min_entry = 0
    for si in range(n_ops + 1):
        tick = stim_tick_per_step[si]
        if tick is not None:
            if tick < _high_water_tick:
                stim_tick_per_step[si] = _high_water_tick
            else:
                _high_water_tick = tick

        entries = stim_lines_per_step[si]
        if entries:
            min_entry = min(entries)
            if min_entry < _high_water_min_entry:
                if si in _directly_matched:
                    # Ground-truth provenance — keep entries as-is,
                    # but do NOT regress the high-water mark.
                    pass
                else:
                    # Interpolated / heuristic — filter entries
                    filtered = {e for e in entries
                                if e >= _high_water_min_entry}
                    if not filtered:
                        clamped_tick = stim_tick_per_step[si] or 0
                        clamped_block = max(0, min(clamped_tick,
                                                   n_blocks - 1))
                        block_entries = set(
                            block_gate_entries[clamped_block])
                        filtered = {e for e in block_entries
                                    if e >= _high_water_min_entry}
                        if not filtered:
                            filtered = {_high_water_min_entry}
                    stim_lines_per_step[si] = filtered
                    new_min = min(filtered)
                    if new_min > _high_water_min_entry:
                        _high_water_min_entry = new_min
            else:
                _high_water_min_entry = min_entry

    return True


# =============================================================================
# Animation legend helper
# =============================================================================

def _draw_animation_legend(ax: Any, ion_roles: Optional[Dict[int, str]]) -> None:
    """Draw role + infrastructure legend on the architecture panel."""
    from matplotlib.lines import Line2D
    from .constants import TRAP_FILL, JUNCTION_FILL, _ROLE_COLORS

    handles = []
    if ion_roles:
        seen: Set[str] = set()
        role_info = {
            "D": ("Data", "#42A5F5"),
            "M": ("Measure", "#EF5350"),
            "A": ("Ancilla", "#EF5350"),
            "S": ("Spectator", "#78909C"),
            "C": ("Cooling", "#66BB6A"),
            "P": ("Prep", "#AB47BC"),
        }
        for role in ion_roles.values():
            k = role[0].upper()
            if k not in seen and k in role_info:
                seen.add(k)
                name, color = role_info[k]
                h = Line2D([0], [0], marker="o", color="w",
                           markerfacecolor=_ROLE_COLORS.get(role, color),
                           markersize=11, label=name, linestyle="None")
                handles.append(h)
    # Infrastructure
    handles.append(Line2D([0], [0], marker="s", color="w",
                          markerfacecolor=TRAP_FILL, markersize=13,
                          markeredgecolor="#5C6BC0", markeredgewidth=1.2,
                          label="Trap", linestyle="None"))
    handles.append(Line2D([0], [0], marker="s", color="w",
                          markerfacecolor=JUNCTION_FILL, markersize=12,
                          markeredgecolor="#E65100", markeredgewidth=1.2,
                          label="Junction", linestyle="None"))
    if handles:
        ax.legend(handles=handles, loc="upper center",
                  bbox_to_anchor=(0.5, -0.02), ncol=len(handles),
                  fontsize=13, framealpha=0.85, fancybox=True,
                  edgecolor="#ccc")


# =============================================================================
# Operation-list formatting for the side panel
# =============================================================================

_SHORT_OP_NAMES = {
    "TwoQubitMSGate": "MS", "MSGate": "MS", "GateSwap": "SWAP",
    "OneQubitGate": "ROT", "XRotation": "RX", "YRotation": "RY",
    "SingleQubitGate": "ROT",
    "Measurement": "MEAS", "MeasurementOperation": "MEAS",
    "QubitReset": "RESET", "ResetOperation": "RESET",
    "Split": "SPLIT", "Merge": "MERGE",
    "JunctionCrossing": "CROSS", "Move": "MOVE",
    "CrystalRotation": "ROTATE", "SympatheticCooling": "COOL",
    "ReconfigurationStep": "RECONFIG",
}


def _format_op_list(op: Any) -> List[str]:
    """Format an operation (possibly parallel) into display strings.

    Returns a list like ``['SPLIT I2 I7', 'RX I3', 'MS I4 I5']``.
    """
    cls = type(op).__name__
    if cls == "ParallelOperation":
        lines: List[str] = []
        for sub in getattr(op, "operations", []):
            lines.extend(_format_op_list(sub))
        return lines
    # Leaf operation
    name = _SHORT_OP_NAMES.get(cls, cls.upper())
    ions = _extract_ions_from_leaf(op)
    ion_str = " ".join(f"I{i}" for i in sorted(ions))
    return [f"{name} {ion_str}".strip()]


# =============================================================================
# V-SWAP detection for proportional timing
# =============================================================================

def _has_v_swap(op: Any) -> bool:
    """Check if an operation involves a vertical junction crossing (V-swap)."""
    cls = type(op).__name__
    if cls == "JunctionCrossing":
        return True
    if cls == "ParallelOperation":
        return any(_has_v_swap(s) for s in getattr(op, "operations", []))
    if hasattr(op, "direction"):
        d = str(getattr(op, "direction", "")).lower()
        return d in ("v", "vertical", "up", "down")
    return False


# =============================================================================
# Main entry point
# =============================================================================

def animate_transport(
    arch: Any,
    operations: Sequence[Any],
    interval: int = 200,
    show_labels: bool = True,
    ion_roles: Optional[Dict[int, str]] = None,
    interp_frames: int = 5,
    gate_hold_frames: int = 3,
    stim_circuit: Any = None,
    ion_idx_remap: Optional[Dict[int, int]] = None,
    physical_to_logical: Optional[Dict[int, int]] = None,
    figsize: Optional[Tuple[float, float]] = None,
    title_prefix: str = "",
    repeat: bool = False,
    native_circuit: Any = None,
    show_trap_labels: bool = True,
    end_hold_frames: int = 0,
    simple_ion_labels: bool = False,
    margin: Optional[float] = None,
) -> Any:
    """Animate step-by-step ion transport with interpolated motion.

    Runs all operations to build position snapshots, then produces an
    animation with smooth interpolation, laser beams, progress bar,
    stim sidebar, and stim SVG timeslice panel.

    Parameters
    ----------
    arch : QCCDArch | QCCDWiseArch
        Architecture instance. Must have ``refreshGraph()``.
    operations : list
        Sequence of ParallelOperation / gate / transport operations.
    interval : int
        Milliseconds between frames (default 200).
    show_labels : bool
        Whether to show ion index labels.
    ion_roles : dict | None
        Mapping ``ion_idx -> role`` (D/M/P/C).
    interp_frames : int
        Frames for smooth interpolation per step (0 = legacy 1-frame-per-op).
    gate_hold_frames : int
        Extra frames to hold on gate steps (laser visible).
    stim_circuit : stim.Circuit | None
        If provided, stim sidebar + SVG timeslice panels are shown.
    figsize : tuple | None
        Figure size in inches (auto-detected if None).
    title_prefix : str
        Prefix for frame titles.
    repeat : bool
        Whether the animation should loop.
    end_hold_frames : int
        Extra frames to hold the final state before looping (default 0).
    simple_ion_labels : bool
        If True, ion labels use simple ``I{idx}`` format instead of
        the full ``D(I{idx})`` role-prefixed format (default False).
    margin : float | None
        Override for the axis margin around the grid (default None
        = auto-detect: 2.0 for WISE, 1.2 otherwise).

    Returns
    -------
    matplotlib.animation.FuncAnimation
    """
    _require_matplotlib()
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation as _FA
    from matplotlib.gridspec import GridSpec

    n_ops = len(operations)
    if n_ops == 0:
        warnings.warn("No operations provided to animate_transport()")

    # Try to import calibration data for physical timing
    _gate_time = dict(GATE_TIME_US)
    try:
        from ..utils.physics import DEFAULT_CALIBRATION as _CAL
        _gate_time["ms"] = getattr(_CAL, "ms_gate_time", 40e-6) * 1e6
        _gate_time["rotation"] = getattr(_CAL, "single_qubit_gate_time", 5e-6) * 1e6
        _gate_time["measure"] = getattr(_CAL, "measurement_time", 100e-6) * 1e6
        _gate_time["reset"] = getattr(_CAL, "reset_time", 5e-6) * 1e6
    except Exception:
        pass

    # ------------------------------------------------------------------
    # 1. Pre-compute snapshots (destructive — runs all operations)
    # ------------------------------------------------------------------
    snapshots, active_ions_per_step, gate_kind_per_step, \
        is_transport_step, labels_per_step, step_duration_us, \
        trap_occupancy_per_step = \
        _build_snapshots(arch, operations)

    # ------------------------------------------------------------------
    # 2. Compute frame counts per step
    # ------------------------------------------------------------------
    if interp_frames <= 0:
        interp_frames = 1
        gate_hold_frames = 0

    step_nframes: List[int] = []
    us_per_frame = H_PASS_TIME_US / max(1, interp_frames)
    for si in range(1, n_ops + 1):
        gk = gate_kind_per_step[si]
        if gk is not None:
            step_nframes.append(interp_frames + gate_hold_frames)
        else:
            # Proportional timing: V-swaps take ~2.4x longer
            if _has_v_swap(operations[si - 1]):
                nf = max(interp_frames, int(V_PASS_TIME_US / us_per_frame))
            else:
                nf = interp_frames
            step_nframes.append(nf)

    # --- Auto-scale to prevent excessive frame counts ---
    _MAX_FRAMES = 1200
    _raw_total = sum(step_nframes) + 1
    if _raw_total > _MAX_FRAMES:
        _shrink = _MAX_FRAMES / _raw_total
        step_nframes = [max(2, int(nf * _shrink)) for nf in step_nframes]

    # Cumulative frame starts (step 0 = initial state = frame 0)
    cum_frames: List[int] = [1]  # frame 0 = initial state (1 frame)
    for nf in step_nframes:
        cum_frames.append(cum_frames[-1] + nf)
    total_frames = cum_frames[-1] + max(0, end_hold_frames)

    # ------------------------------------------------------------------
    # 3. Compute axis bounds from geometry
    # ------------------------------------------------------------------
    _mro = _mro_names(arch)
    is_wise = "QCCDWiseArch" in _mro or "WISEArchitecture" in _mro

    geom = read_trap_geometries(arch)
    all_x: List[float] = []
    all_y: List[float] = []
    for t in geom["traps"]:
        px, py = t["pos"]
        tw, th = t["width"], t["height"]
        all_x.extend([px - tw / 2, px + tw / 2])
        all_y.extend([py - th / 2, py + th / 2])
    for j in geom["junctions"]:
        jx, jy = j["pos"]
        all_x.append(jx)
        all_y.append(jy)
    # Also include all ion positions from all snapshots
    for snap in snapshots:
        for ix, iy in snap.values():
            all_x.append(ix)
            all_y.append(iy)

    _margin = margin if margin is not None else (2.0 if is_wise else 1.2)
    x_lo = min(all_x) - _margin if all_x else -5
    x_hi = max(all_x) + _margin if all_x else 10
    y_lo = min(all_y) - _margin if all_y else -5
    y_hi = max(all_y) + _margin * 1.5 if all_y else 10

    # ------------------------------------------------------------------
    # 4. Stim mapping (if stim_circuit provided)
    # ------------------------------------------------------------------
    has_stim = stim_circuit is not None
    sidebar_entries = []
    tick_svg_cache: Dict[int, Any] = {}
    stim_lines_per_step: List[Optional[Set[int]]] = [None] * (n_ops + 1)
    stim_tick_per_step: List[Optional[int]] = [None] * (n_ops + 1)
    stim_n_ticks = 0
    has_svg = False

    if has_stim:
        sidebar_entries, tick_svg_cache, stim_lines_per_step, \
            stim_tick_per_step, stim_n_ticks = \
            _build_stim_mapping(
                stim_circuit, n_ops, gate_kind_per_step,
                active_ions_per_step=active_ions_per_step,
                ion_idx_remap=ion_idx_remap,
                physical_to_logical=physical_to_logical,
                operations=operations,
            )
        has_svg = bool(tick_svg_cache)

    # ------------------------------------------------------------------
    # 5. Figure + GridSpec layout  (adaptive to grid + circuit size)
    # ------------------------------------------------------------------
    n_q = stim_circuit.num_qubits if stim_circuit is not None else 0

    if figsize is None:
        dx = (max(all_x) - min(all_x)) if all_x else 10
        dy = (max(all_y) - min(all_y)) if all_y else 8
        if is_wise:
            base_w = max(6, min(8, dx * 0.35 + 2))
            base_h = max(6, min(8, dy * 0.35 + 2))
        else:
            base_w = max(5, min(7, dx * 0.30 + 2))
            base_h = max(5, min(7, dy * 0.30 + 2))

        if has_stim and has_svg:
            # Timeslice + sidebar add width; sidebar is compact (6 lines)
            ts_extra_w = max(3, min(5, 2.5 + n_q * 0.03))
            figsize = (base_w + ts_extra_w + 1, max(base_h, 6))
        elif has_stim:
            figsize = (base_w + 2, base_h)
        else:
            figsize = (base_w, base_h)

    # Use high DPI so the (smaller) figure has enough pixel resolution.
    # This keeps text large on screen because the browser doesn't need to
    # shrink a giant image to fit the notebook output cell.
    effective_dpi = DPI  # 120 from constants

    # Raise the jshtml embed limit to 80 MB for large animations
    plt.rcParams['animation.embed_limit'] = 80

    fig = plt.figure(figsize=figsize, dpi=effective_dpi)
    plt.close(fig)

    # GridSpec: timeslice ratio grows with qubit count
    _ts_ratio = min(2.8, max(1.5, 1.2 + n_q * 0.015))
    # Sidebar is narrow – only 6 lines of text
    _sb_ratio = 0.85

    if has_stim and has_svg:
        gs = GridSpec(1, 3, figure=fig,
                      width_ratios=[3.0, _ts_ratio, _sb_ratio],
                      wspace=0.04)
        ax = fig.add_subplot(gs[0, 0])
        ax_topo = fig.add_subplot(gs[0, 1])
        ax_sidebar = fig.add_subplot(gs[0, 2])
    elif has_stim:
        gs = GridSpec(1, 2, figure=fig, width_ratios=[3.0, _sb_ratio],
                      wspace=0.04)
        ax = fig.add_subplot(gs[0, 0])
        ax_topo = None
        ax_sidebar = fig.add_subplot(gs[0, 1])
    else:
        ax = fig.add_subplot(111)
        ax_topo = None
        ax_sidebar = None

    # Remove whitespace around the figure – pack panels edge-to-edge
    fig.subplots_adjust(left=0.02, right=0.98, top=0.95, bottom=0.03)

    # ------------------------------------------------------------------
    # 6. Physical timing
    # ------------------------------------------------------------------
    total_time_us = sum(step_duration_us)

    # ------------------------------------------------------------------
    # 7. Cache for skip-redraw optimisation
    # ------------------------------------------------------------------
    _cache: Dict[str, Any] = {"prev_op_idx": -999, "prev_t": -1.0,
                               "prev_tick": -1}

    # Track failed frames
    failed_frames: List[Tuple[int, str]] = []

    # ------------------------------------------------------------------
    # 8. _update function
    # ------------------------------------------------------------------
    def _update(frame: int) -> None:
        import bisect as _bisect

        op_idx = -1
        t = 0.0

        if frame > 0 and n_ops > 0:
            # Find which step this frame belongs to
            op_idx = _bisect.bisect_right(cum_frames, frame) - 1
            op_idx = max(0, min(op_idx, n_ops - 1))
            sf = cum_frames[op_idx]
            nf = step_nframes[op_idx]
            sub = frame - sf

            if sub < interp_frames:
                t = _ease((sub + 1) / interp_frames)
            else:
                t = 1.0  # hold phase

            # Skip identical redraws — MUST be before ax.clear() to
            # prevent blank frames (flicker) during gate-hold phase.
            if (op_idx == _cache["prev_op_idx"]
                    and t == _cache["prev_t"]):
                return

        _cache["prev_op_idx"] = op_idx
        _cache["prev_t"] = t

        # --- Clear and set up axes ---
        ax.clear()
        ax.set_facecolor(BG_COLOR)
        ax.axis("off")
        ax.set_xlim(x_lo, x_hi)
        ax.set_ylim(y_lo, y_hi)
        ax.set_aspect("equal")

        # --- Draw static grid ---
        gk = gate_kind_per_step[op_idx + 1] if op_idx >= 0 and op_idx + 1 < len(gate_kind_per_step) else None
        # Use pre-step trap occupancy during interpolation, post-step at t=1
        if op_idx < 0:
            _frame_occ = trap_occupancy_per_step[0] if trap_occupancy_per_step else None
        elif t >= 1.0:
            _step_idx = min(op_idx + 1, len(trap_occupancy_per_step) - 1)
            _frame_occ = trap_occupancy_per_step[_step_idx]
        else:
            _step_idx = min(op_idx, len(trap_occupancy_per_step) - 1)
            _frame_occ = trap_occupancy_per_step[_step_idx]
        if is_wise:
            _draw_wise_grid(ax, arch, gate_kind=gk, show_trap_labels=show_trap_labels)
        else:
            _draw_qccd_grid(ax, arch, gate_kind=gk, show_trap_labels=show_trap_labels,
                            ion_count_override=_frame_occ)

        if frame == 0:
            # Initial state
            _draw_ions_full(
                ax, snapshots[0], set(),
                ion_roles=ion_roles,
                gate_kind=None,
                show_labels=show_labels,
                show_laser=False,
                ion_idx_remap=ion_idx_remap,
                physical_to_logical=physical_to_logical,
                per_ion_gate_kind=None,
                simple_ion_labels=simple_ion_labels,
            )
            title_str = "Initial configuration"
            if n_ops == 0:
                title_str = "Initial configuration (no ops)"
        else:
            prev_snap = snapshots[op_idx]
            next_snap = snapshots[op_idx + 1]
            active = active_ions_per_step[op_idx + 1]
            gk = gate_kind_per_step[op_idx + 1]
            is_xp = is_transport_step[op_idx + 1]

            # Interpolate positions
            interp_pos: Dict[int, Tuple[float, float]] = {}
            trails: Dict[int, Tuple[Tuple[float, float], Tuple[float, float]]] = {}
            all_idxs = set(prev_snap) | set(next_snap)

            for idx in all_idxs:
                px, py = prev_snap.get(idx, next_snap.get(idx, (0, 0)))
                nx_, ny_ = next_snap.get(idx, prev_snap.get(idx, (0, 0)))
                ix = px + (nx_ - px) * t
                iy = py + (ny_ - py) * t
                interp_pos[idx] = (ix, iy)

                if idx in active and (abs(px - nx_) > 0.1 or abs(py - ny_) > 0.1):
                    trails[idx] = ((px, py), (ix, iy))

            # Gate background tint
            if gk and gk in GATE_BG_TINTS:
                tint_c, tint_a = GATE_BG_TINTS[gk]
                from matplotlib.patches import FancyBboxPatch
                bg = FancyBboxPatch(
                    (x_lo, y_lo), x_hi - x_lo, y_hi - y_lo,
                    boxstyle="round,pad=0",
                    facecolor=tint_c, alpha=tint_a,
                    edgecolor="none", zorder=0)
                ax.add_patch(bg)

            # Draw ions
            # For gate steps the laser should stay on for the entire
            # duration (no flicker). Only suppress during transport.
            # Also keep laser on during transition from gate→gate steps
            # (consecutive gates without transport) to avoid flicker.
            show_laser = (gk is not None) and (not is_xp)
            if not show_laser and is_xp and op_idx > 0:
                # Check if previous AND next steps are both gates —
                # if so, keep laser on during the transitional transport
                prev_gk = gate_kind_per_step[op_idx] if op_idx < len(gate_kind_per_step) else None
                next_gk = gate_kind_per_step[op_idx + 2] if op_idx + 2 < len(gate_kind_per_step) else None
                if prev_gk is not None and next_gk is not None:
                    show_laser = True
                    gk = prev_gk  # use the previous gate kind for tint
            _pigk = _extract_per_ion_gate_kind(operations[op_idx])
            _ms_pairs = _extract_ms_pairs(operations[op_idx])
            _draw_ions_full(
                ax, interp_pos, active,
                ion_roles=ion_roles,
                gate_kind=gk if show_laser else None,
                show_labels=show_labels,
                show_laser=show_laser,
                ion_idx_remap=ion_idx_remap,
                physical_to_logical=physical_to_logical,
                per_ion_gate_kind=_pigk,
                ms_pairs=_ms_pairs,
                simple_ion_labels=simple_ion_labels,
            )

            # Transport arrows
            if is_xp and t < 1.0 and trails:
                _draw_transport_arrows(ax, trails)

            # Title
            if is_xp:
                kind = "[MOVE]"
            elif gk == "ms":
                kind = "[MS GATE]"
            elif gk == "rotation":
                kind = "[ROTATION]"
            elif gk == "measure":
                kind = "[MEASURE]"
            elif gk == "reset":
                kind = "[RESET]"
            else:
                kind = "[GATE]"
            op_label = labels_per_step[op_idx + 1]
            title_str = f"{kind} {op_label}"

        # Title (short — just the prefix or kind)
        ax.set_title(
            title_prefix or title_str,
            fontsize=20, fontweight="bold", pad=12)

        # Ops list text box (right side)
        if frame > 0:
            op_lines = _format_op_list(operations[op_idx])
            if op_lines:
                ops_txt = "\n".join(op_lines[:12])  # cap at 12 entries
                if len(op_lines) > 12:
                    ops_txt += f"\n… +{len(op_lines)-12} more"
                ax.text(0.98, 0.88, ops_txt,
                        transform=ax.transAxes,
                        fontsize=14, fontfamily="monospace",
                        va="top", ha="right", color="#222",
                        bbox=dict(facecolor="white", alpha=0.90,
                                  edgecolor="#bbb",
                                  boxstyle="round,pad=0.35"),
                        zorder=100)

        # --- Progress bar ---
        if n_ops > 0:
            elapsed_us = sum(step_duration_us[:op_idx + 2]) if op_idx >= 0 else 0.0
            progress = min(1.0, elapsed_us / total_time_us) if total_time_us > 0 else (
                (op_idx + 1) / n_ops if op_idx >= 0 else 0.0
            )
            draw_progress_bar(
                ax, progress, elapsed_us, total_time_us,
                step=(op_idx + 1) if op_idx >= 0 else 0,
                total_steps=n_ops,
            )

        # --- Sidebar ---
        if ax_sidebar is not None:
            if has_stim and sidebar_entries:
                hl = stim_lines_per_step[op_idx + 1] if op_idx >= 0 and op_idx + 1 < len(stim_lines_per_step) else set()
                center = min(hl) if hl else 0
                draw_sidebar(
                    ax_sidebar, sidebar_entries,
                    highlight_lines=hl or set(),
                    center_idx=center,
                    title="Stim Circuit",
                )
            else:
                draw_ops_sidebar(
                    ax_sidebar,
                    labels_per_step,
                    gate_kind_per_step,
                    is_transport_step,
                    current_step=(op_idx + 1) if op_idx >= 0 else 0,
                    title="Timeslice  (steps)",
                )

        # --- Stim SVG timeslice ---
        if ax_topo is not None and has_svg:
            cur_tick = 0
            if op_idx >= 0 and op_idx + 1 < len(stim_tick_per_step):
                ct = stim_tick_per_step[op_idx + 1]
                if ct is not None:
                    cur_tick = ct
            cur_tick = max(0, min(cur_tick, max(tick_svg_cache.keys()) if tick_svg_cache else 0))

            # Always redraw — ax is cleared each frame anyway
            _cache["prev_tick"] = cur_tick
            svg_data = tick_svg_cache.get(cur_tick)
            if svg_data:
                draw_stim_svg(
                    ax_topo, svg_data,
                    title=f"Timeslice  TICK {cur_tick}/{stim_n_ticks}")
            else:
                ax_topo.clear()
                ax_topo.set_facecolor(SIDEBAR_BG)
                ax_topo.axis("off")
                ax_topo.text(
                    0.5, 0.5,
                    f"TICK {cur_tick}\n(no SVG data)",
                    transform=ax_topo.transAxes,
                    ha="center", va="center",
                    fontsize=13, color="#999")

        # --- Legend ---
        _draw_animation_legend(ax, ion_roles)

    # ------------------------------------------------------------------
    # 9. Create animation
    # ------------------------------------------------------------------
    anim = _FA(
        fig, _update,
        frames=total_frames,
        interval=interval,
        repeat=repeat,
    )

    # Attach metadata for diagnostics
    anim._failed_frames = failed_frames
    anim._n_ops = n_ops
    anim._total_frames = total_frames

    return anim


__all__ = ["animate_transport"]
