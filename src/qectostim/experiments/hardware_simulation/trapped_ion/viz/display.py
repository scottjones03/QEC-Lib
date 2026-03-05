"""
Architecture display functions for old/ QCCD simulation.

Uses custom matplotlib rendering (FancyBboxPatch, Circle, laser beams)
ported from the new trapped_ion/viz style, while reading ion positions
from the old arch internals (``arch._manipulationTraps``, ``._junctions``,
``._crossings``) via ``layout.read_ion_positions()`` and
``layout.read_trap_geometries()``.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Set, Tuple

from .constants import (
    BG_COLOR,
    CROSSING_COLOR,
    CROSSING_VERT_COLOR,
    DPI,
    HIGHLIGHT_ION,
    ION_BELOW_FONT,
    ION_EDGE_WIDTH,
    ION_FONT,
    ION_OUTLINE,
    ION_RADIUS,
    ION_SPACING_RATIO,
    JUNCTION_EDGE,
    JUNCTION_FILL,
    JUNCTION_SIDE,
    LEGEND_FONT,
    QUBIT_DEFAULT,
    SPACING,
    STROKE,
    STROKE_THIN,
    TRAP_EDGE,
    TRAP_FILL,
    TRAP_LABEL_FONT,
    TRAP_PAD_X,
    TRAP_PAD_Y,
    _ROLE_COLORS,
    LASER_MS,
    LASER_ROTATION,
    LASER_MEASURE,
    LASER_RESET,
    GATE_BG_TINTS,
    TRAP_GATE_FILL,
    TRAP_GATE_EDGE,
    TRAP_PAD_ROUND,
    TRAP_LW,
    JUNCTION_PAD_ROUND,
    JUNCTION_LW,
    _OLD_TRANSPORT_CLASSES,
    _OLD_MS_CLASSES,
    _OLD_1Q_CLASSES,
    _OLD_MEAS_CLASSES,
    _OLD_RESET_CLASSES,
)

try:
    import matplotlib.pyplot as plt
    import matplotlib.patheffects as path_effects
    from matplotlib.patches import Circle, FancyBboxPatch
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


# =============================================================================
# Helpers  (kept for backward compat — reconfig.py imports them)
# =============================================================================

def _mro_names(obj: Any) -> Tuple[str, ...]:
    """Return class names from the MRO of *obj*."""
    return tuple(cls.__name__ for cls in type(obj).__mro__)


def _require_matplotlib() -> None:
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib is required.  pip install matplotlib")


def _ion_role_letter(ion_idx: int,
                     ion_roles: Optional[Dict[int, str]] = None) -> str:
    """Return single-char role letter, e.g. 'D', 'M', 'S'."""
    if ion_roles and ion_idx in ion_roles:
        return ion_roles[ion_idx][0].upper()
    return "D"


def _draw_ion(ax: Any, x: float, y: float, idx: int,
              highlight_set: Set[int],
              ion_roles: Optional[Dict[int, str]],
              show_label: bool = True,
              radius: float = 0.38,
              zorder: int = 10) -> None:
    """Draw a single ion (kept for reconfig.py backward-compat)."""
    color = HIGHLIGHT_ION if idx in highlight_set else (
        _ROLE_COLORS.get(ion_roles[idx], QUBIT_DEFAULT)
        if ion_roles and idx in ion_roles else QUBIT_DEFAULT
    )
    circ = Circle(
        (x, y), radius,
        facecolor=color, edgecolor=ION_OUTLINE,
        linewidth=ION_EDGE_WIDTH, zorder=zorder,
    )
    ax.add_patch(circ)
    role_ch = _ion_role_letter(idx, ion_roles)
    ax.text(
        x, y, role_ch,
        fontsize=ION_FONT, ha="center", va="center",
        fontweight="bold", color="white", zorder=zorder + 1,
        path_effects=[path_effects.withStroke(linewidth=2.5, foreground=color)],
    )
    if show_label:
        ax.text(
            x, y - radius - 0.16, str(idx),
            fontsize=ION_BELOW_FONT, ha="center", va="top",
            fontweight="bold", zorder=zorder + 1,
            path_effects=STROKE_THIN, color="#333",
        )


def _ion_color(ion_idx: int, highlight_set: Set[int],
               ion_roles: Optional[Dict[int, str]] = None) -> str:
    """Return fill colour (kept for backward-compat)."""
    if ion_idx in highlight_set:
        return HIGHLIGHT_ION
    if ion_roles and ion_idx in ion_roles:
        return _ROLE_COLORS.get(ion_roles[ion_idx], QUBIT_DEFAULT)
    return QUBIT_DEFAULT


# =============================================================================
# Gate kind / transport detection helpers
# =============================================================================


def _is_transport_single(op: Any) -> bool:
    """Check if a *single* (non-parallel) operation is transport-related."""
    cls_name = type(op).__name__
    if cls_name in _OLD_TRANSPORT_CLASSES:
        return True
    if hasattr(op, "source_zone") or hasattr(op, "_crossing"):
        return True
    for base in type(op).__mro__:
        if base.__name__ in ("TransportOperation", "CrystalOperation"):
            return True
    return False


def _is_transport(op: Any) -> bool:
    """Check if an operation is *purely* transport (no gate sub-ops).

    For a ``ParallelOperation``, returns ``True`` only if **all** sub-ops
    are transport.  Mixed steps (transport + gate) return ``False`` so
    the gate gets visualised.
    """
    cls_name = type(op).__name__
    if cls_name == "ParallelOperation":
        sub_ops = getattr(op, "operations", [])
        return bool(sub_ops) and all(_is_transport(s) for s in sub_ops)
    return _is_transport_single(op)


# Gate-kind priority for mixed ParallelOperations:
#   ms > measure > reset > rotation
_GATE_KIND_PRIORITY = {"ms": 4, "measure": 3, "reset": 2, "rotation": 1}


def _gate_kind_single(op: Any) -> Optional[str]:
    """Return 'ms', 'rotation', 'measure', 'reset' or None for a leaf op."""
    cls = type(op).__name__
    if cls == "MeasurementOperation":
        return "measure"
    if cls == "ResetOperation":
        return "reset"
    if cls in _OLD_MS_CLASSES:
        return "ms"
    if cls in _OLD_1Q_CLASSES:
        return "rotation"
    if cls in _OLD_MEAS_CLASSES:
        return "measure"
    if cls in _OLD_RESET_CLASSES:
        return "reset"
    # Name-based fallback
    name = (getattr(op, "gate_name", None)
            or getattr(op, "name", None)
            or getattr(op, "label", None)
            or "")
    if not name:
        gate_obj = getattr(op, "gate", None)
        if gate_obj is not None:
            name = getattr(gate_obj, "name", "") or ""
    name_lower = str(name).lower()
    if name_lower in ("ms", "xx", "zz", "cx", "cnot", "cz"):
        return "ms"
    if name_lower in ("m", "mz", "mx", "measure", "measurement"):
        return "measure"
    if name_lower in ("h", "s", "t", "rx", "ry", "rz", "sdg", "tdg",
                       "x", "y", "z", "r", "reset"):
        return "rotation"
    # Check operation_type attribute
    op_type = getattr(op, "operation_type", None)
    if op_type is not None:
        ot = str(op_type).lower()
        if "two" in ot or "2q" in ot:
            return "ms"
        if "one" in ot or "1q" in ot:
            return "rotation"
    return None


def _gate_kind(op: Any) -> Optional[str]:
    """Return the dominant gate kind for an operation.

    For ``ParallelOperation``, recurses into all sub-ops and returns
    the highest-priority gate kind found (ms > measure > reset > rotation).
    Returns ``None`` if no gate sub-ops are present (pure transport).
    """
    cls = type(op).__name__
    if cls == "ParallelOperation":
        sub_ops = getattr(op, "operations", [])
        best: Optional[str] = None
        best_pri = -1
        for sub in sub_ops:
            gk = _gate_kind(sub)
            if gk is not None:
                pri = _GATE_KIND_PRIORITY.get(gk, 0)
                if pri > best_pri:
                    best = gk
                    best_pri = pri
        return best
    return _gate_kind_single(op)


def _extract_ions_from_leaf(op: Any) -> Set[int]:
    """Extract ion indices from a single (non-parallel) operation."""
    active: Set[int] = set()
    for attr in ("ions", "qubits", "_ions", "_qubits",
                 "targets", "ion_indices"):
        val = getattr(op, attr, None)
        if val:
            for item in val:
                if isinstance(item, int):
                    active.add(item)
                else:
                    idx = getattr(item, "idx", None)
                    if idx is not None:
                        active.add(int(idx))
    # Singular .ion / ._ion
    for attr in ("ion", "_ion"):
        single = getattr(op, attr, None)
        if single is not None:
            idx = getattr(single, "idx", None)
            if idx is not None:
                active.add(int(idx))
            elif isinstance(single, int):
                active.add(single)
    return active


def _extract_active_ions(op: Any) -> Set[int]:
    """Extract ion indices for highlighting / laser beams.

    For ``ParallelOperation`` containing *both* gate and transport sub-ops,
    returns only the ions from gate sub-ops so that laser beams and glow
    are applied exclusively to gate-involved ions.  For pure-transport
    steps, all involved ions are returned.
    """
    cls = type(op).__name__
    if cls == "ParallelOperation":
        sub_ops = getattr(op, "operations", [])
        gate_ions: Set[int] = set()
        transport_ions: Set[int] = set()
        for sub in sub_ops:
            sub_ions = _extract_active_ions(sub)  # recurse for nested parallel
            if _gate_kind(sub) is not None:
                gate_ions |= sub_ions
            else:
                transport_ions |= sub_ions
        # Prefer gate ions when mixed; fall back to transport-only ions
        return gate_ions if gate_ions else transport_ions
    return _extract_ions_from_leaf(op)


def _extract_per_ion_gate_kind(op: Any) -> Dict[int, str]:
    """Return ``{ion_idx: gate_kind}`` for all ions in an operation.

    For ``ParallelOperation``, recurses into sub-ops so each ion gets
    the correct gate type (e.g. rotation vs reset in the same step).
    """
    result: Dict[int, str] = {}
    cls = type(op).__name__
    if cls == "ParallelOperation":
        for sub in getattr(op, "operations", []):
            result.update(_extract_per_ion_gate_kind(sub))
    else:
        gk = _gate_kind_single(op)
        if gk:
            for idx in _extract_ions_from_leaf(op):
                result[idx] = gk
    return result


def _extract_ms_pairs(op: Any) -> List[Tuple[int, int]]:
    """Return MS gate ion pairs from an operation tree.

    For ``ParallelOperation``, recurses into sub-ops and collects every
    leaf MS gate as a ``(ion_a, ion_b)`` pair.  This allows the display
    to draw connecting beams per pair even when multiple MS gates fire
    simultaneously.
    """
    pairs: List[Tuple[int, int]] = []
    cls = type(op).__name__
    if cls == "ParallelOperation":
        for sub in getattr(op, "operations", []):
            pairs.extend(_extract_ms_pairs(sub))
    else:
        gk = _gate_kind_single(op)
        if gk == "ms":
            ions = sorted(_extract_ions_from_leaf(op))
            if len(ions) == 2:
                pairs.append((ions[0], ions[1]))
    return pairs


# =============================================================================
# Custom QCCD grid drawing  (replaces arch.display() / networkx)
# =============================================================================

def _draw_qccd_grid(
    ax: Any,
    arch: Any,
    highlight_traps: Optional[Set[int]] = None,
    gate_kind: Optional[str] = None,
    ion_count_override: Optional[Dict[int, int]] = None,
    show_trap_labels: bool = True,
) -> None:
    """Draw the static QCCD infrastructure using FancyBboxPatch.

    Reads trap/junction/crossing geometry from ``arch`` internals
    via ``read_trap_geometries()``.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
    arch : QCCDArch / QCCDWiseArch
    highlight_traps : set of trap indices to highlight (gate zone)
    gate_kind : str or None
        If highlighting, which gate kind for the tint colour.
    ion_count_override : {trap_idx: n_ions} or None
        Override ion counts for dynamic trap sizing during animation.
    """
    from .layout import read_trap_geometries

    geom = read_trap_geometries(arch)
    if highlight_traps is None:
        highlight_traps = set()

    # --- Crossings (edges) first so traps draw on top ---
    for edge in geom["crossings"]:
        sx, sy = edge["src_pos"]
        tx, ty = edge["tgt_pos"]
        # Detect vertical vs horizontal
        is_vert = abs(tx - sx) < abs(ty - sy)
        ls = "--" if is_vert else "-"
        color = "#B0BEC5" if is_vert else "#90A4AE"
        ax.plot([sx, tx], [sy, ty],
                color=color, linewidth=1.4,
                linestyle=ls, solid_capstyle="round",
                zorder=1, alpha=0.6)

    # --- Traps ---
    for trap in geom["traps"]:
        px, py = trap["pos"]
        tidx = trap["idx"]

        # Dynamic sizing: use override ion count if provided
        if ion_count_override and tidx in ion_count_override:
            ni = ion_count_override[tidx]
            inner = max(0, (ni - 1)) * SPACING * ION_SPACING_RATIO
            if trap.get("is_horizontal", True):
                tw = max(1.6, inner + 2 * TRAP_PAD_X)
                th = max(0.7, 2 * ION_RADIUS + 2 * TRAP_PAD_Y)
            else:
                tw = max(0.7, 2 * ION_RADIUS + 2 * TRAP_PAD_Y)
                th = max(1.6, inner + 2 * TRAP_PAD_X)
        else:
            tw, th = trap["width"], trap["height"]

        # Gate-zone highlighting
        if tidx in highlight_traps and gate_kind:
            tint_color, tint_alpha = GATE_BG_TINTS.get(
                gate_kind, (TRAP_GATE_FILL, 0.18))
            fc = tint_color
            ec = TRAP_GATE_EDGE
            alpha = 0.92
        else:
            fc = TRAP_FILL
            ec = TRAP_EDGE
            alpha = 0.88

        rect = FancyBboxPatch(
            (px - tw / 2, py - th / 2), tw, th,
            boxstyle=f"round,pad={TRAP_PAD_ROUND}",
            facecolor=fc, edgecolor=ec,
            linewidth=TRAP_LW, zorder=2, alpha=alpha,
        )
        ax.add_patch(rect)
        # Label above trap
        if show_trap_labels:
            ax.text(px, py + th / 2 + 0.12, trap["label"],
                    fontsize=TRAP_LABEL_FONT, ha="center", va="bottom",
                    color=ec, fontweight="bold",
                    path_effects=STROKE_THIN)

    # --- Junctions ---
    for junc in geom["junctions"]:
        px, py = junc["pos"]
        half = JUNCTION_SIDE / 2
        jrect = FancyBboxPatch(
            (px - half, py - half), JUNCTION_SIDE, JUNCTION_SIDE,
            boxstyle=f"round,pad={JUNCTION_PAD_ROUND}",
            facecolor=JUNCTION_FILL, edgecolor=JUNCTION_EDGE,
            linewidth=JUNCTION_LW, zorder=4, alpha=0.85,
        )
        ax.add_patch(jrect)


# =============================================================================
# Custom WISE grid drawing  (for QCCDWiseArch / WISEArchitecture)
# =============================================================================

def _draw_wise_grid(
    ax: Any,
    arch: Any,
    highlight_traps: Optional[Set[int]] = None,
    gate_kind: Optional[str] = None,
    show_trap_labels: bool = True,
) -> None:
    """Draw the static WISE infrastructure using FancyBboxPatch.

    Uses the WISE layout extractor from ``layout.py`` to compute trap,
    junction, and edge positions from ``arch.m``, ``arch.n``, ``arch.k``.
    Visual style matches the ``trapped_ion/viz/visualization.py`` WISE renderer.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
    arch : QCCDWiseArch / WISEArchitecture
    highlight_traps : set of trap indices to highlight (gate zone)
    gate_kind : str or None
        If highlighting, which gate kind for the tint colour.
    """
    from .layout import _extract_wise_layout

    if highlight_traps is None:
        highlight_traps = set()

    # Compute layout geometry
    ion_sp = SPACING * ION_SPACING_RATIO
    layout = _extract_wise_layout(arch, ion_sp)

    # --- Crossing edges first (so traps paint on top) ---
    for edge in layout.edges:
        ls = "--" if edge.dashed else "-"
        color = CROSSING_VERT_COLOR if edge.dashed else CROSSING_COLOR
        lw = 1.2 if edge.dashed else 1.5
        ax.plot([edge.x0, edge.x1], [edge.y0, edge.y1],
                color=color, linewidth=lw,
                linestyle=ls, solid_capstyle="round",
                zorder=1, alpha=0.6 if edge.dashed else 0.8)

    # --- Trap rectangles ---
    for tidx, trap in enumerate(layout.traps):
        tw, th = trap.width, trap.height

        if tidx in highlight_traps and gate_kind:
            tint_color, tint_alpha = GATE_BG_TINTS.get(
                gate_kind, (TRAP_GATE_FILL, 0.18))
            fc = tint_color
            ec = TRAP_GATE_EDGE
            alpha = 0.92
        else:
            fc = TRAP_FILL
            ec = TRAP_EDGE
            alpha = 0.88

        rect = FancyBboxPatch(
            (trap.cx - tw / 2, trap.cy - th / 2), tw, th,
            boxstyle=f"round,pad={TRAP_PAD_ROUND}",
            facecolor=fc, edgecolor=ec,
            linewidth=TRAP_LW, zorder=2, alpha=alpha,
        )
        ax.add_patch(rect)
        if show_trap_labels:
            ax.text(trap.cx, trap.cy + th / 2 + 0.15, trap.label,
                    fontsize=TRAP_LABEL_FONT, ha="center", va="bottom",
                    color=ec, fontweight="bold",
                    path_effects=STROKE_THIN)

    # --- Junctions ---
    half = JUNCTION_SIDE / 2
    for junc in layout.junctions:
        jrect = FancyBboxPatch(
            (junc.cx - half, junc.cy - half), JUNCTION_SIDE, JUNCTION_SIDE,
            boxstyle=f"round,pad={JUNCTION_PAD_ROUND}",
            facecolor=JUNCTION_FILL, edgecolor=JUNCTION_EDGE,
            linewidth=JUNCTION_LW, zorder=4, alpha=0.85,
        )
        ax.add_patch(jrect)


def _draw_ions_full(
    ax: Any,
    positions: Dict[int, Tuple[float, float]],
    active_set: Set[int],
    ion_roles: Optional[Dict[int, str]] = None,
    gate_kind: Optional[str] = None,
    show_labels: bool = True,
    show_laser: bool = True,
    ion_idx_remap: Optional[Dict[int, int]] = None,
    physical_to_logical: Optional[Dict[int, int]] = None,
    per_ion_gate_kind: Optional[Dict[int, str]] = None,
    ms_pairs: Optional[List[Tuple[int, int]]] = None,
    simple_ion_labels: bool = False,
) -> None:
    """Draw all ions at their current positions with role colours and lasers.

    Ported from trapped_ion/viz/visualization.py ``_draw_ions()``.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
    positions : {ion_idx: (x, y)}
    active_set : set of ion indices currently active (highlighted)
    ion_roles : {ion_idx: role_str} or None
    gate_kind : 'ms' | 'rotation' | 'measure' | 'reset' | None
    show_labels : show index labels below ions
    show_laser : draw laser beams on active ions
    ion_idx_remap : optional {physical_idx: display_idx}
    physical_to_logical : optional {physical_idx: logical_qubit}
    """
    ion_r = ION_RADIUS
    laser_offset = (2 * ION_RADIUS + 2 * TRAP_PAD_Y) * 0.7

    # Per-ion active highlight colours (differentiate rotation vs reset etc.)
    _ACTIVE_GATE_COLORS = {
        "ms": HIGHLIGHT_ION,        # yellow
        "rotation": "#CE93D8",     # light purple
        "reset": "#80DEEA",        # light cyan
        "measure": "#A5D6A7",      # light green
    }

    for idx, (ix, iy) in sorted(positions.items()):
        is_active = idx in active_set
        if is_active:
            ion_gk = (per_ion_gate_kind or {}).get(idx, gate_kind)
            color = _ACTIVE_GATE_COLORS.get(ion_gk, HIGHLIGHT_ION)
            edge_c = "#E65100"
            lw = 2.2
        elif ion_roles and idx in ion_roles:
            color = _ROLE_COLORS.get(ion_roles[idx], QUBIT_DEFAULT)
            edge_c = ION_OUTLINE
            lw = 1.2
        else:
            color = QUBIT_DEFAULT
            edge_c = ION_OUTLINE
            lw = 1.2

        circ = Circle(
            (ix, iy), ion_r,
            facecolor=color, edgecolor=edge_c,
            linewidth=lw, zorder=10,
        )
        ax.add_patch(circ)
        role_ch = _ion_role_letter(idx, ion_roles)
        ax.text(ix, iy, role_ch,
                fontsize=ION_FONT, ha="center", va="center",
                fontweight="bold", color="white", zorder=11,
                path_effects=[path_effects.withStroke(
                    linewidth=2.5, foreground=color)])
        if show_labels:
            # Format: D(I{ion_idx}Q{qubit_idx}) or M(I{ion_idx}Q{qubit_idx})
            # simple_ion_labels: just "I{idx}" (no role prefix)
            if simple_ion_labels:
                label_str = f"I{idx}"
            elif physical_to_logical:
                qubit_id = physical_to_logical.get(idx)
                if qubit_id is not None:
                    label_str = f"{role_ch}(I{idx}Q{qubit_id})"
                else:
                    label_str = f"{role_ch}(I{idx})"
            elif ion_idx_remap:
                display_idx = ion_idx_remap.get(idx, idx)
                label_str = f"{role_ch}(I{idx}Q{display_idx})"
            else:
                label_str = f"{role_ch}(I{idx})"
            ax.text(ix, iy - ion_r - 0.15, label_str,
                    fontsize=ION_BELOW_FONT, ha="center", va="top",
                    fontweight="bold", color="#333", zorder=11,
                    path_effects=STROKE_THIN)

    # --- Laser beams for gate steps ---
    if not show_laser or not active_set:
        return

    beam_color_map = {
        "ms": LASER_MS,
        "rotation": LASER_ROTATION,
        "measure": LASER_MEASURE,
        "reset": LASER_RESET,
    }
    _GATE_LABEL_MAP = {
        "ms": "MS", "rotation": "ROT", "measure": "MEAS", "reset": "RESET",
    }

    # Build per-ion gate-kind map for differentiated laser colours
    ion_gk_map: Dict[int, str] = {}
    for ion_idx in active_set:
        if ion_idx not in positions:
            continue
        if per_ion_gate_kind and ion_idx in per_ion_gate_kind:
            ion_gk_map[ion_idx] = per_ion_gate_kind[ion_idx]
        elif gate_kind:
            ion_gk_map[ion_idx] = gate_kind

    if not ion_gk_map:
        return

    # Group ions by gate kind so each group gets its own beam colour
    from collections import defaultdict
    kind_groups: Dict[str, List[Tuple[int, Tuple[float, float]]]] = defaultdict(list)
    for idx_g, gk_g in ion_gk_map.items():
        kind_groups[gk_g].append((idx_g, positions[idx_g]))

    for gk_g, group_ions in kind_groups.items():
        beam_color = beam_color_map.get(gk_g)
        if not beam_color:
            continue

        if gk_g == "ms" and len(group_ions) >= 2:
            # Two converging beams from above + connecting beam
            for _, (ix, iy) in group_ions:
                bx_start = ix
                by_start = iy + laser_offset
                ax.plot([bx_start, ix], [by_start, iy + ion_r],
                        color=beam_color, linewidth=6.0, alpha=0.85,
                        solid_capstyle="round", zorder=8)
                glow = Circle((ix, iy), ion_r * 2.8,
                              facecolor=beam_color, alpha=0.40,
                              edgecolor="none", zorder=9)
                ax.add_patch(glow)
                halo = Circle((ix, iy), ion_r * 4.0,
                              facecolor=beam_color, alpha=0.12,
                              edgecolor="none", zorder=7)
                ax.add_patch(halo)

            # Bug 2 fix: draw connecting beams per MS pair instead of
            # only when there are exactly 2 MS ions in the batch.
            _drawn_pairs: List[Tuple[int, int]] = []
            if ms_pairs:
                # Use explicit pair info from the operation tree
                for ia, ib in ms_pairs:
                    if ia in positions and ib in positions:
                        _drawn_pairs.append((ia, ib))
            elif len(group_ions) == 2:
                # Fallback: only 2 MS ions → they form the pair
                _drawn_pairs.append((group_ions[0][0], group_ions[1][0]))

            for ia, ib in _drawn_pairs:
                x1, y1 = positions[ia]
                x2, y2 = positions[ib]
                ax.plot([x1, x2], [y1, y2],
                        color=beam_color, linewidth=5.0, alpha=0.70,
                        solid_capstyle="round", zorder=8)
                ax.plot([x1, x2], [y1, y2],
                        color=beam_color, linewidth=12.0, alpha=0.18,
                        solid_capstyle="round", zorder=6)

            for _, (ix, iy) in group_ions:
                ring = Circle((ix, iy), ion_r * 1.1,
                              facecolor="none", edgecolor=beam_color,
                              linewidth=2.5, zorder=11)
                ax.add_patch(ring)

            ms_x = sum(p[0] for _, p in group_ions) / len(group_ions)
            ms_y = max(p[1] for _, p in group_ions) + laser_offset * 1.2
            ax.text(ms_x, ms_y, "MS",
                    fontsize=12, ha="center", va="bottom",
                    fontweight="bold", color=beam_color, zorder=12,
                    path_effects=[path_effects.withStroke(
                        linewidth=3, foreground="white")])
        else:
            # Single downward beam per ion
            for _, (ix, iy) in group_ions:
                bx_start = ix
                by_start = iy + laser_offset * 1.5
                ax.plot([bx_start, ix], [by_start, iy + ion_r],
                        color=beam_color, linewidth=5.0, alpha=0.85,
                        solid_capstyle="round", zorder=8)
                ax.plot([bx_start, ix], [by_start, iy + ion_r],
                        color=beam_color, linewidth=14.0, alpha=0.30,
                        solid_capstyle="round", zorder=7)
                glow = Circle((ix, iy), ion_r * 2.5,
                              facecolor=beam_color, alpha=0.45,
                              edgecolor="none", zorder=9)
                ax.add_patch(glow)
                halo = Circle((ix, iy), ion_r * 4.0,
                              facecolor=beam_color, alpha=0.20,
                              edgecolor="none", zorder=7)
                ax.add_patch(halo)
                ring = Circle((ix, iy), ion_r * 1.1,
                              facecolor="none", edgecolor=beam_color,
                              linewidth=2.5, zorder=11)
                ax.add_patch(ring)

            lbl = _GATE_LABEL_MAP.get(gk_g, "GATE")
            lx = sum(p[0] for _, p in group_ions) / len(group_ions)
            ly = max(p[1] for _, p in group_ions) + laser_offset * 1.7
            ax.text(lx, ly, lbl,
                    fontsize=11, ha="center", va="bottom",
                    fontweight="bold", color=beam_color, zorder=12,
                    path_effects=[path_effects.withStroke(
                        linewidth=3, foreground="white")])


def _draw_transport_arrows(
    ax: Any,
    trails: Dict[int, Tuple[Tuple[float, float], Tuple[float, float]]],
) -> None:
    """Draw transport arrows showing ion motion.

    Parameters
    ----------
    trails : {ion_idx: (old_xy, new_xy)}
        Maps each moving ion to its (start, end) position pair.
    """
    if not trails:
        return

    drawn_pairs: Set[Tuple[int, int]] = set()
    trail_keys = list(trails.keys())

    for ti, idx_t in enumerate(trail_keys):
        old_xy, new_xy = trails[idx_t]
        if old_xy == new_xy:
            continue

        # Check for swap partner
        partner = None
        for tj in range(ti + 1, len(trail_keys)):
            idx_t2 = trail_keys[tj]
            old2, new2 = trails[idx_t2]
            # Swap: A's old ~ B's new and vice versa
            if (abs(old_xy[0] - new2[0]) < 1.0
                    and abs(old_xy[1] - new2[1]) < 1.0
                    and abs(new_xy[0] - old2[0]) < 1.0
                    and abs(new_xy[1] - old2[1]) < 1.0):
                partner = idx_t2
                break

        if partner is not None and (partner, idx_t) not in drawn_pairs:
            drawn_pairs.add((idx_t, partner))
            drawn_pairs.add((partner, idx_t))
            old2_p, _ = trails[partner]
            ax.plot([old_xy[0], old2_p[0]], [old_xy[1], old2_p[1]],
                    color="#E65100", linewidth=3.0, alpha=0.55,
                    solid_capstyle="round", zorder=9, linestyle="--")
            for sid in (idx_t, partner):
                so, sn = trails[sid]
                ax.annotate(
                    "", xy=sn, xytext=so,
                    arrowprops=dict(arrowstyle="-|>",
                                    color="#E65100", lw=2.5, alpha=0.7,
                                    connectionstyle="arc3,rad=0.08"),
                    zorder=9)
        elif idx_t not in {p for pair in drawn_pairs for p in pair}:
            ax.annotate(
                "", xy=new_xy, xytext=old_xy,
                arrowprops=dict(arrowstyle="-|>",
                                color="#E65100", lw=2.0, alpha=0.6,
                                connectionstyle="arc3,rad=0.15"),
                zorder=9)


# =============================================================================
# Legend helper
# =============================================================================

def _draw_role_legend(ax: Any, ion_roles: Optional[Dict[int, str]],
                     legend_loc: str = "lower right") -> None:
    """Draw a small legend showing role → colour mapping."""
    if not ion_roles:
        return
    seen: Dict[str, str] = {}
    for role in ion_roles.values():
        key = role[0].upper()
        if key not in seen:
            full = {"D": "Data", "M": "Measure/Ancilla",
                    "A": "Ancilla", "S": "Spectator",
                    "C": "Cooling"}.get(key, role.capitalize())
            seen[key] = full
    if not seen:
        return

    handles = []
    labels = []
    for letter, full_name in seen.items():
        color = _ROLE_COLORS.get(
            next((r for r in ion_roles.values() if r[0].upper() == letter), ""),
            QUBIT_DEFAULT,
        )
        h = Circle((0, 0), 0.1, facecolor=color, edgecolor=ION_OUTLINE)
        handles.append(h)
        labels.append(f"{letter} = {full_name}")

    if handles:
        if legend_loc == "below":
            ax.legend(handles, labels, loc="upper center",
                      bbox_to_anchor=(0.5, -0.02), ncol=len(handles),
                      fontsize=LEGEND_FONT, framealpha=0.8,
                      edgecolor="#ccc", fancybox=True)
        else:
            ax.legend(handles, labels, loc=legend_loc,
                      fontsize=LEGEND_FONT, framealpha=0.8,
                      edgecolor="#ccc", fancybox=True)


# =============================================================================
# Public entry point — custom rendering
# =============================================================================

def display_architecture(
    arch: Any,
    fig: Any = None,
    ax: Any = None,
    title: str = "",
    show_junctions: bool = True,
    show_edges: bool = True,
    show_ions: bool = True,
    show_labels: bool = True,
    show_trap_labels: bool = True,
    highlight_qubits: Optional[list] = None,
    ion_roles: Optional[Dict[int, str]] = None,
    show_legend: bool = True,
    figsize: Optional[Tuple[float, float]] = None,
    ion_idx_remap: Optional[Dict[int, int]] = None,
    physical_to_logical: Optional[Dict[int, int]] = None,
    legend_loc: str = "lower right",
    margin: Optional[float] = None,
    simple_ion_labels: bool = False,
) -> Tuple[Any, Any]:
    """Display a QCCD architecture topology.

    Uses custom matplotlib rendering (FancyBboxPatch, Circle, laser beams)
    instead of delegating to ``arch.display()``.

    Parameters
    ----------
    arch : QCCDArch | QCCDWiseArch
        Architecture instance to render.
    fig, ax : optional
        Existing figure / axes to draw into.
    title : str
        Plot title.
    show_labels : bool
        Whether to show ion / trap / junction labels.
    figsize : tuple | None
        Explicit ``(width, height)`` in inches.

    Returns
    -------
    (Figure, Axes)
    """
    _require_matplotlib()
    from .layout import read_ion_positions

    if fig is None or ax is None:
        if figsize is None:
            _mro = _mro_names(arch)
            is_wise = "QCCDWiseArch" in _mro or "WISEArchitecture" in _mro
            if is_wise:
                m = getattr(arch, "col_groups", getattr(arch, "m", 2))
                n = getattr(arch, "rows", getattr(arch, "n", 2))
                figsize = (max(12, m * 6.0), max(6, n * 3.0))
            else:
                # Adaptive: compute from actual geometry bounds
                from .layout import read_trap_geometries as _rtg
                _geom = _rtg(arch)
                _ax_all, _ay_all = [], []
                for _t in _geom["traps"]:
                    _px, _py = _t["pos"]
                    _tw, _th = _t["width"], _t["height"]
                    _ax_all.extend([_px - _tw / 2, _px + _tw / 2])
                    _ay_all.extend([_py - _th / 2, _py + _th / 2])
                for _j in _geom["junctions"]:
                    _jx, _jy = _j["pos"]
                    _ax_all.append(_jx); _ay_all.append(_jy)
                if _ax_all and _ay_all:
                    _dx = max(_ax_all) - min(_ax_all)
                    _dy = max(_ay_all) - min(_ay_all)
                    figsize = (max(10, min(20, _dx * 1.0 + 5)),
                               max(8, min(16, _dy * 1.0 + 4)))
                else:
                    figsize = (14, 10)
        fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=DPI)

    # Ensure graph is up to date
    if hasattr(arch, "refreshGraph"):
        try:
            arch.refreshGraph()
        except Exception:
            pass

    ax.set_facecolor(BG_COLOR)
    ax.axis("off")

    # Draw grid — dispatch based on architecture type
    _mro_all = _mro_names(arch)
    _is_wise_arch = ("QCCDWiseArch" in _mro_all
                     or "WISEArchitecture" in _mro_all)
    if _is_wise_arch:
        _draw_wise_grid(ax, arch, show_trap_labels=show_trap_labels)
    else:
        _draw_qccd_grid(ax, arch, show_trap_labels=show_trap_labels)

    # Draw ions
    if show_ions:
        positions = read_ion_positions(arch)
        highlight_set = set(highlight_qubits) if highlight_qubits else set()
        _draw_ions_full(
            ax, positions, highlight_set,
            ion_roles=ion_roles,
            gate_kind=None,
            show_labels=show_labels,
            show_laser=False,
            ion_idx_remap=ion_idx_remap,
            physical_to_logical=physical_to_logical,
            simple_ion_labels=simple_ion_labels,
        )

    # Auto-compute axis limits from geometry
    from .layout import read_trap_geometries
    geom = read_trap_geometries(arch)
    all_x, all_y = [], []
    for t in geom["traps"]:
        px, py = t["pos"]
        tw, th = t["width"], t["height"]
        all_x.extend([px - tw / 2, px + tw / 2])
        all_y.extend([py - th / 2, py + th / 2])
    for j in geom["junctions"]:
        jx, jy = j["pos"]
        all_x.append(jx)
        all_y.append(jy)
    if all_x and all_y:
        _margin = margin if margin is not None else 0.5
        ax.set_xlim(min(all_x) - _margin, max(all_x) + _margin)
        ax.set_ylim(min(all_y) - _margin, max(all_y) + _margin * 1.2)
    ax.set_aspect("equal")

    if title:
        ax.set_title(title, fontsize=16, fontweight="bold", pad=14)

    if show_legend:
        _draw_role_legend(ax, ion_roles, legend_loc=legend_loc)

    return fig, ax
