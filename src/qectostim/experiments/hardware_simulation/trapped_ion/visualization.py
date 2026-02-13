# src/qectostim/experiments/hardware_simulation/trapped_ion/visualization.py
"""
High-quality visualization for trapped-ion architectures.

Provides matplotlib-based rendering of:
- WISE architecture with trap-block / junction / crossing topology
- QCCD architecture topology (traps, junctions, crossings, ion positions)
- Linear-chain architecture with segmented trap zones
- Step-by-step transport animation with smooth interpolation
- WISE reconfiguration / swap-schedule phases

The WISE renderer mirrors the old ``processCircuitWiseArch`` layout:
  * Traps at *even*-column grid positions ``(2*col, row)``
  * Junctions at *odd*-column positions ``(2*col+1, row)``
  * Crossings wired between adjacent traps and junctions

Usage
-----
>>> from qectostim.experiments.hardware_simulation.trapped_ion.visualization import (
...     display_architecture,
...     animate_transport,
...     visualize_reconfiguration,
... )
>>> fig, ax = display_architecture(arch, title="My WISE grid")
>>> plt.show()
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
    TYPE_CHECKING,
)

try:
    import matplotlib
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.patches import Ellipse, FancyBboxPatch, FancyArrowPatch, Circle
    from matplotlib.collections import PatchCollection
    from matplotlib.lines import Line2D
    import matplotlib.patheffects as path_effects
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False


def _mro_names(obj) -> Tuple[str, ...]:
    """Return class names from the MRO (method resolution order).

    Used instead of isinstance() to avoid stale module-cache issues
    while still supporting subclass dispatch (e.g. AugmentedGridArchitecture
    → QCCDArchitecture).
    """
    return tuple(cls.__name__ for cls in type(obj).__mro__)


if TYPE_CHECKING:
    from matplotlib.figure import Figure
    from matplotlib.axes import Axes
    from matplotlib.animation import FuncAnimation
    from qectostim.experiments.hardware_simulation.trapped_ion.architecture import (
        TrappedIonArchitecture,
        QCCDArchitecture,
        WISEArchitecture,
        LinearChainArchitecture,
        QCCDGraph,
        QCCDNode,
        Ion,
        ManipulationTrap,
        Junction,
        Crossing,
    )


# =============================================================================
# Styling Constants  (tuned for publication-quality output)
# =============================================================================

DPI = 180
ION_RADIUS = 0.38
JUNCTION_SIDE = 0.62
TRAP_PAD_X = 0.65
TRAP_PAD_Y = 0.50
ION_MARKER_SIZE = 800

FONT_SIZE = 15
ION_FONT = 13           # font for ion role letter *inside* circle
ION_BELOW_FONT = 9      # font for index label *below* circle
TRAP_LABEL_FONT = 11
JUNCTION_FONT = 10
LEGEND_FONT = 11
GATE_FONT = 14
INFO_FONT = 10

TRAP_LINEWIDTH = 2.2
EDGE_LINEWIDTH = 2.2
ION_EDGE_WIDTH = 1.6
CROSSING_LW = 2.0
HIGHLIGHT_LW = 2.8

SPACING = 3.8
ION_SPACING_RATIO = 0.92

# Ion role colours — distinct, colour-blind-friendly palette
DATA_COLOR = "#2979FF"       # vivid blue
ANCILLA_COLOR = "#E53935"    # vivid red
SPECTATOR_COLOR = "#9E9E9E"  # grey
COOLING_COLOR = "#43A047"    # green
HIGHLIGHT_ION = "#FFD600"
HIGHLIGHT_BG = "#FFFDE7"

QUBIT_DEFAULT = DATA_COLOR
JUNCTION_FILL = "#FFA726"
JUNCTION_EDGE = "#E65100"
TRAP_FILL = "#E8EAF6"
TRAP_EDGE = "#5C6BC0"
TRAP_GATE_FILL = "#FFF9C4"
TRAP_GATE_EDGE = "#F9A825"
CROSSING_COLOR = "#78909C"
CROSSING_VERT_COLOR = "#B0BEC5"
ION_OUTLINE = "#212121"

_ROLE_COLORS: Dict[str, str] = {
    "D": DATA_COLOR, "data": DATA_COLOR,
    "M": ANCILLA_COLOR, "ancilla": ANCILLA_COLOR, "A": ANCILLA_COLOR,
    "P": SPECTATOR_COLOR, "spectator": SPECTATOR_COLOR, "S": SPECTATOR_COLOR,
    "C": COOLING_COLOR, "cooling": COOLING_COLOR,
}

_ROLE_LEGEND: List[Tuple[str, str, str]] = [
    ("D", "Data qubit", DATA_COLOR),
    ("M", "Ancilla / Meas", ANCILLA_COLOR),
    ("P", "Spectator", SPECTATOR_COLOR),
    ("C", "Cooling ion", COOLING_COLOR),
]

_STROKE = [path_effects.withStroke(linewidth=3.5, foreground="white")]
_STROKE_THIN = [path_effects.withStroke(linewidth=2.5, foreground="white")]


# =============================================================================
# Helpers
# =============================================================================

def _ion_color(ion_idx, highlight_set, ion_roles=None):
    """Return the fill colour for *ion_idx*."""
    if ion_idx in highlight_set:
        return HIGHLIGHT_ION
    if ion_roles and ion_idx in ion_roles:
        return _ROLE_COLORS.get(ion_roles[ion_idx], QUBIT_DEFAULT)
    return QUBIT_DEFAULT


def _ion_role_letter(ion_idx, ion_roles=None):
    """Return single-char role letter, e.g. 'D', 'M', 'S'."""
    if ion_roles and ion_idx in ion_roles:
        return ion_roles[ion_idx][0].upper()
    return "D"


def _ion_label(ion_idx, ion_roles=None):
    """Return a short label string for below the ion, e.g. '3' or 'D3'."""
    if ion_roles and ion_idx in ion_roles:
        letter = ion_roles[ion_idx][0].upper()
        return f"{letter}{ion_idx}"
    return str(ion_idx)


def _require_matplotlib():
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib is required.  pip install matplotlib")


def _require_networkx():
    if not HAS_NETWORKX:
        raise ImportError("networkx is required.  pip install networkx")


def _draw_ion(ax, x, y, idx, highlight_set, ion_roles,
              show_label=True, radius=ION_RADIUS, zorder=10):
    """Draw a single ion: filled circle with role letter inside,
    numeric index below."""
    color = _ion_color(idx, highlight_set, ion_roles)
    circ = Circle(
        (x, y), radius,
        facecolor=color, edgecolor=ION_OUTLINE,
        linewidth=ION_EDGE_WIDTH, zorder=zorder,
    )
    ax.add_patch(circ)
    # Role letter centred inside the circle (white, bold)
    role_ch = _ion_role_letter(idx, ion_roles)
    ax.text(
        x, y, role_ch,
        fontsize=ION_FONT, ha="center", va="center",
        fontweight="bold", color="white", zorder=zorder + 1,
        path_effects=[path_effects.withStroke(linewidth=2.5, foreground=color)],
    )
    # Numeric index below the circle
    if show_label:
        ax.text(
            x, y - radius - 0.16, str(idx),
            fontsize=ION_BELOW_FONT, ha="center", va="top",
            fontweight="bold", zorder=zorder + 1,
            path_effects=_STROKE_THIN, color="#333",
        )


def _add_legend(ax, ion_roles=None):
    """Add colour-legend for ion roles + structural elements."""
    handles = []
    seen = set()
    for key, label, color in _ROLE_LEGEND:
        if ion_roles is None:
            if key == "D" and key not in seen:
                handles.append(Line2D(
                    [0], [0], marker="o", color="w", markerfacecolor=color,
                    markeredgecolor=ION_OUTLINE, markersize=12, label=label,
                ))
                seen.add(key)
        else:
            present = any(
                v.startswith(key) or v == label.split()[0].lower()
                for v in ion_roles.values()
            )
            if present and key not in seen:
                handles.append(Line2D(
                    [0], [0], marker="o", color="w", markerfacecolor=color,
                    markeredgecolor=ION_OUTLINE, markersize=12, label=label,
                ))
                seen.add(key)
    handles.append(Line2D(
        [0], [0], marker="s", color="w", markerfacecolor=JUNCTION_FILL,
        markeredgecolor=JUNCTION_EDGE, markersize=12, label="Junction",
    ))
    handles.append(Line2D(
        [0], [0], marker="s", color="w", markerfacecolor=TRAP_FILL,
        markeredgecolor=TRAP_EDGE, markersize=14, label="Trap segment",
    ))
    if handles:
        ax.legend(
            handles=handles, loc="upper right",
            fontsize=LEGEND_FONT, framealpha=0.94,
            edgecolor="#bbb", fancybox=True,
            borderpad=0.7, handletextpad=0.6,
        )


# =============================================================================
# Unified layout data + shared renderer for WISE / QCCD / AugGrid / Networked
# =============================================================================

@dataclass
class _TrapInfo:
    """Position and content of one trap segment."""
    cx: float
    cy: float
    width: float
    height: float
    label: str
    ion_indices: List[int]          # ordered ion indices inside this trap
    is_horizontal: bool = True


@dataclass
class _JunctionInfo:
    """Position and label of one junction node."""
    cx: float
    cy: float
    label: str


@dataclass
class _EdgeInfo:
    """A connection line between two layout points."""
    x0: float
    y0: float
    x1: float
    y1: float
    label: str = ""
    dashed: bool = False


@dataclass
class _GraphLayout:
    """Architecture-agnostic layout data for the shared renderer."""
    traps: List[_TrapInfo]
    junctions: List[_JunctionInfo]
    edges: List[_EdgeInfo]
    ion_positions: Dict[int, Tuple[float, float]]   # ion_idx → (x, y)
    info_lines: List[str]
    auto_title: str


def _extract_wise_layout(arch, ion_sp: float) -> _GraphLayout:
    """Build a ``_GraphLayout`` from a WISE architecture."""
    rows = arch.rows
    k = arch.ions_per_segment
    m = arch.col_groups
    total_cols = arch.total_columns

    trap_inner_w = (k - 1) * ion_sp
    trap_w = trap_inner_w + 2 * TRAP_PAD_X
    trap_h = 2 * ION_RADIUS + 2 * TRAP_PAD_Y
    junc_gap = SPACING * 1.2
    block_pitch = trap_w + junc_gap
    row_pitch = SPACING * 3.5

    def _trap_xy(b, r):
        return (b * block_pitch, r * row_pitch)

    def _junc_xy(b, r):
        tx, _ = _trap_xy(b, r)
        return (tx + trap_w / 2 + junc_gap / 2, r * row_pitch)

    def _ion_xy(b, r, slot):
        tx, ty = _trap_xy(b, r)
        return (tx - trap_inner_w / 2 + slot * ion_sp, ty)

    traps: List[_TrapInfo] = []
    junctions: List[_JunctionInfo] = []
    edges: List[_EdgeInfo] = []
    ion_positions: Dict[int, Tuple[float, float]] = {}

    # Traps & ions
    for b in range(m):
        for r in range(rows):
            cx, cy = _trap_xy(b, r)
            ion_ids = []
            for slot in range(k):
                ion_idx = r * total_cols + b * k + slot
                ix, iy = _ion_xy(b, r, slot)
                ion_positions[ion_idx] = (ix, iy)
                ion_ids.append(ion_idx)
            traps.append(_TrapInfo(
                cx=cx, cy=cy, width=trap_w, height=trap_h,
                label=f"T({b},{r}) k={k}",
                ion_indices=ion_ids, is_horizontal=True,
            ))

    # Junctions
    for b in range(m - 1):
        for r in range(rows):
            jx, jy = _junc_xy(b, r)
            junctions.append(_JunctionInfo(jx, jy, f"J({b},{r})"))

    # Horizontal crossing edges (trap → junction → next trap)
    for r in range(rows):
        for b in range(m - 1):
            tx1, ty1 = _trap_xy(b, r)
            jx, jy = _junc_xy(b, r)
            tx2, ty2 = _trap_xy(b + 1, r)
            re = tx1 + trap_w / 2
            le = tx2 - trap_w / 2
            edges.append(_EdgeInfo(
                re, ty1, jx - JUNCTION_SIDE / 2, jy,
                label=f"E({b}\u2194{b+1},r{r})",
            ))
            edges.append(_EdgeInfo(
                jx + JUNCTION_SIDE / 2, jy, le, ty2,
            ))

    # Vertical dashed links between junctions
    for r in range(rows - 1):
        for b in range(m - 1):
            jx1, jy1 = _junc_xy(b, r)
            jx2, jy2 = _junc_xy(b, r + 1)
            edges.append(_EdgeInfo(
                jx1, jy1 + JUNCTION_SIDE / 2,
                jx2, jy2 - JUNCTION_SIDE / 2,
                dashed=True,
            ))

    # Info
    info_lines = [
        f"m={m}  n={rows}  k={k}",
        f"Ions: {arch.num_qubits}   Traps: {m*rows}",
        f"Junctions: {(m-1)*rows}",
    ]
    auto_title = (
        f"WISE  m={m} \u00d7 n={rows} \u00d7 k={k}  [{arch.num_qubits} ions]"
    )
    return _GraphLayout(
        traps=traps, junctions=junctions, edges=edges,
        ion_positions=ion_positions, info_lines=info_lines,
        auto_title=auto_title,
    )


def _extract_qccd_layout(arch, ion_sp: float) -> _GraphLayout:
    """Build a ``_GraphLayout`` from any QCCD-graph architecture."""
    graph = arch.qccd_graph

    traps: List[_TrapInfo] = []
    junctions: List[_JunctionInfo] = []
    edges: List[_EdgeInfo] = []
    ion_positions: Dict[int, Tuple[float, float]] = {}

    # --- nodes → traps / junctions ---
    for node in graph.nodes.values():
        _ntype = type(node).__name__
        px, py = node.position

        if "Trap" in _ntype:
            node_ions = list(getattr(node, "ions", []))
            ni = len(node_ions)
            cap = getattr(node, "capacity", ni or 3)
            horiz = getattr(node, "is_horizontal", True)

            inner_span = max(0, (ni - 1)) * ion_sp
            if horiz:
                tw = inner_span + 2 * TRAP_PAD_X
                th = 2 * ION_RADIUS + 2 * TRAP_PAD_Y
            else:
                tw = 2 * ION_RADIUS + 2 * TRAP_PAD_Y
                th = inner_span + 2 * TRAP_PAD_X

            ion_ids = []
            for i, ion in enumerate(node_ions):
                if horiz:
                    ix = px - inner_span / 2 + i * ion_sp
                    iy = py
                else:
                    ix = px
                    iy = py - inner_span / 2 + i * ion_sp
                ion_positions[ion.idx] = (ix, iy)
                ion_ids.append(ion.idx)

            _label = getattr(node, "display_label",
                             getattr(node, "label", f"T{node.idx}"))
            traps.append(_TrapInfo(
                cx=px, cy=py, width=tw, height=th,
                label=f"{_label} k={cap}",
                ion_indices=ion_ids, is_horizontal=horiz,
            ))

        elif "Junction" in _ntype or "Crossing" in _ntype:
            _jlabel = getattr(node, "display_label",
                              getattr(node, "label", f"J{node.idx}"))
            junctions.append(_JunctionInfo(px, py, _jlabel))

    # --- crossing edges ---
    seen_pairs: set = set()
    for crossing in graph.crossings.values():
        si = crossing.source.idx
        ti = crossing.target.idx
        pair = (min(si, ti), max(si, ti))
        if pair in seen_pairs:
            continue
        seen_pairs.add(pair)
        s_node = graph.nodes.get(si)
        t_node = graph.nodes.get(ti)
        if s_node is None or t_node is None:
            continue
        sx, sy = s_node.position
        tx, ty = t_node.position
        edges.append(_EdgeInfo(sx, sy, tx, ty, label=f"E({si}\u2194{ti})"))

    # --- info ---
    n_traps = len(traps)
    n_junctions = len(junctions)
    info_lines = [
        f"Traps: {n_traps}   Junctions: {n_junctions}",
        f"Ions: {arch.num_qubits}   k={getattr(arch, 'ions_per_trap', '?')}",
    ]
    auto_title = f"{type(arch).__name__}  [{arch.num_qubits} ions]"
    return _GraphLayout(
        traps=traps, junctions=junctions, edges=edges,
        ion_positions=ion_positions, info_lines=info_lines,
        auto_title=auto_title,
    )


def _display_trapped_ion_graph(
    layout: _GraphLayout, ax, title: str,
    show_junctions: bool, show_edges: bool,
    show_ions: bool, show_labels: bool,
    highlight_qubits, ion_roles,
):
    """Shared renderer: draws any ``_GraphLayout`` in the unified visual style."""
    highlight_set = set(highlight_qubits or [])

    # ---- 1. Edges -----------------------------------------------------------
    if show_edges:
        for e in layout.edges:
            style: Dict[str, Any] = dict(
                color=CROSSING_VERT_COLOR if e.dashed else CROSSING_COLOR,
                linewidth=CROSSING_LW * (0.85 if e.dashed else 1.0),
                solid_capstyle="round", zorder=1,
                alpha=0.6 if not e.dashed else 0.55,
            )
            if e.dashed:
                style["linestyle"] = "--"
                style["dash_capstyle"] = "round"
            ax.plot([e.x0, e.x1], [e.y0, e.y1], **style)
            if show_labels and e.label:
                mx, my = (e.x0 + e.x1) / 2, (e.y0 + e.y1) / 2
                ax.text(mx, my + 0.6, e.label,
                        fontsize=7, ha="center", va="bottom",
                        color="#607D8B", fontstyle="italic",
                        path_effects=_STROKE_THIN)

    # ---- 2. Trap rectangles -------------------------------------------------
    for t in layout.traps:
        rect = FancyBboxPatch(
            (t.cx - t.width / 2, t.cy - t.height / 2), t.width, t.height,
            boxstyle="round,pad=0.20",
            facecolor=TRAP_FILL, edgecolor=TRAP_EDGE,
            linewidth=TRAP_LINEWIDTH, zorder=2, alpha=0.92)
        ax.add_patch(rect)
        ax.text(t.cx, t.cy + t.height / 2 + 0.22, t.label,
                fontsize=TRAP_LABEL_FONT, ha="center", va="bottom",
                color=TRAP_EDGE, fontweight="bold",
                path_effects=_STROKE_THIN)

    # ---- 3. Junctions -------------------------------------------------------
    if show_junctions:
        half = JUNCTION_SIDE / 2
        for j in layout.junctions:
            jrect = FancyBboxPatch(
                (j.cx - half, j.cy - half), JUNCTION_SIDE, JUNCTION_SIDE,
                boxstyle="round,pad=0.07",
                facecolor=JUNCTION_FILL, edgecolor=JUNCTION_EDGE,
                linewidth=1.5, zorder=4, alpha=0.88)
            ax.add_patch(jrect)
            ax.text(j.cx, j.cy, j.label,
                    fontsize=JUNCTION_FONT - 1, ha="center", va="center",
                    color="white", fontweight="bold", zorder=5,
                    path_effects=[path_effects.withStroke(
                        linewidth=2, foreground=JUNCTION_EDGE)])

    # ---- 4. Ions ------------------------------------------------------------
    if show_ions:
        for ion_idx, (ix, iy) in layout.ion_positions.items():
            _draw_ion(ax, ix, iy, ion_idx,
                      highlight_set, ion_roles,
                      show_label=show_labels,
                      radius=ION_RADIUS, zorder=6)

    # ---- 5. Highlight routing path ------------------------------------------
    if highlight_qubits and len(highlight_qubits) >= 2:
        hq = [q for q in highlight_qubits if q in layout.ion_positions]
        if len(hq) >= 2:
            xs = [layout.ion_positions[q][0] for q in hq]
            ys = [layout.ion_positions[q][1] for q in hq]
            cx = (min(xs) + max(xs)) / 2
            cy = (min(ys) + max(ys)) / 2
            ew = max(xs) - min(xs) + ION_RADIUS * 6
            eh = max(ys) - min(ys) + ION_RADIUS * 6
            ellip = Ellipse(
                (cx, cy), max(ew, ION_RADIUS * 6), max(eh, ION_RADIUS * 6),
                edgecolor="#FF6600", facecolor=HIGHLIGHT_BG,
                alpha=0.35, linewidth=HIGHLIGHT_LW, linestyle="--", zorder=3)
            ax.add_patch(ellip)
            ax.plot(xs, ys, color="#FF6600", linewidth=2.0,
                    linestyle=":", alpha=0.80, zorder=3,
                    marker="o", markersize=4)

    # ---- 6. Info panel ------------------------------------------------------
    info = list(layout.info_lines)
    if ion_roles:
        role_counts: Dict[str, int] = {}
        for v in ion_roles.values():
            ch = v[0].upper()
            role_counts[ch] = role_counts.get(ch, 0) + 1
        parts = [f"{rk}:{rv}" for rk, rv in sorted(role_counts.items())]
        info.append("Roles: " + "  ".join(parts))
    ax.text(0.01, 0.01, "\n".join(info),
            transform=ax.transAxes,
            fontsize=INFO_FONT, va="bottom", ha="left",
            fontfamily="monospace",
            bbox=dict(facecolor="white", alpha=0.88,
                      edgecolor="#bbb", boxstyle="round,pad=0.5"),
            zorder=100)

    # ---- 7. Title & limits --------------------------------------------------
    if title:
        ax.set_title(title, fontsize=FONT_SIZE + 2, fontweight="bold", pad=14)
    else:
        ax.set_title(layout.auto_title,
                     fontsize=FONT_SIZE + 1, fontweight="bold", pad=14)
    ax.set_aspect("equal")
    ax.set_facecolor("#FAFBFE")
    ax.axis("off")

    all_x: List[float] = []
    all_y: List[float] = []
    for t in layout.traps:
        all_x.extend([t.cx - t.width / 2, t.cx + t.width / 2])
        all_y.extend([t.cy - t.height / 2, t.cy + t.height / 2])
    for j in layout.junctions:
        all_x.append(j.cx)
        all_y.append(j.cy)
    pad = SPACING * 0.9
    if all_x and all_y:
        ax.set_xlim(min(all_x) - pad, max(all_x) + pad)
        ax.set_ylim(min(all_y) - pad - 0.8, max(all_y) + pad + 1.0)


# =============================================================================
# Public entry point
# =============================================================================

def display_architecture(arch, fig=None, ax=None, title="",
                         show_junctions=True, show_edges=True,
                         show_ions=True, show_labels=True,
                         highlight_qubits=None, ion_roles=None,
                         show_legend=True, figsize=None):
    """Display a trapped-ion architecture topology.

    Parameters
    ----------
    arch : TrappedIonArchitecture
        Architecture instance to render.
    fig, ax : optional
        Existing figure / axes to draw into.
    title : str
        Plot title.  Auto-generated if empty.
    highlight_qubits : list[int] | None
        Ion indices to highlight with gold circles & routing path.
    ion_roles : dict[int, str] | None
        Mapping ``ion_idx -> role`` where role is D/M/P/C.
    show_legend : bool
        Whether to show the colour legend.
    figsize : tuple | None
        Explicit ``(width, height)`` in inches.

    Returns
    -------
    (Figure, Axes)
    """
    _require_matplotlib()

    # Use MRO-aware name matching instead of isinstance() to avoid
    # stale module-cache issues while still supporting subclass
    # dispatch (e.g. AugmentedGridArchitecture → QCCDArchitecture).
    _cls_name = type(arch).__name__
    _mro = _mro_names(arch)

    if fig is None or ax is None:
        if figsize is None:
            if _cls_name == "WISEArchitecture":
                m, n = arch.col_groups, arch.rows
                # Scale generously so nothing overlaps
                figsize = (max(9, m * 5.0), max(6.5, n * 4.0))
            elif _cls_name == "LinearChainArchitecture":
                figsize = (max(10, arch.num_qubits * 1.4), 4.5)
            else:
                figsize = (15, 11)
        fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=DPI)

    if _cls_name == "WISEArchitecture":
        _display_wise_grid(arch, fig, ax, title,
                           show_ions, show_labels, highlight_qubits, ion_roles)
    elif "QCCDArchitecture" in _mro and hasattr(arch, "qccd_graph"):
        _display_qccd(arch, fig, ax, title,
                      show_junctions, show_edges, show_ions, show_labels,
                      highlight_qubits, ion_roles)
    elif _cls_name == "LinearChainArchitecture":
        _display_linear_chain(arch, fig, ax, title,
                              show_ions, show_labels, highlight_qubits, ion_roles)
    else:
        _display_generic(arch, fig, ax, title)

    if show_legend:
        _add_legend(ax, ion_roles)
    return fig, ax


# =============================================================================
# WISE Grid Renderer
# =============================================================================

def _display_wise_grid(arch, fig, ax, title,
                       show_ions, show_labels, highlight_qubits, ion_roles=None):
    """Render WISE m × n × k architecture with traps, junctions, crossings.

    Each trap segment is drawn as a rounded rectangle containing ``k``
    ion circles.  Junctions are labelled with their ``J(b,r)`` index.
    Horizontal crossings are labelled ``E(b↔b+1,r)`` and vertical
    links are shown as dashed lines.  An info panel summarises grid
    dimensions and role counts.
    """
    rows = arch.rows
    k = arch.ions_per_segment
    m = arch.col_groups
    total_cols = arch.total_columns
    highlight_set = set(highlight_qubits or [])

    # --- coordinate system (generous spacing to avoid overlaps) ---
    ion_sp = SPACING * ION_SPACING_RATIO
    trap_inner_w = (k - 1) * ion_sp
    trap_w = trap_inner_w + 2 * TRAP_PAD_X
    trap_h = 2 * ION_RADIUS + 2 * TRAP_PAD_Y
    junc_gap = SPACING * 1.2           # wider gap so junction label fits
    block_pitch = trap_w + junc_gap
    row_pitch = SPACING * 3.5          # more vertical space

    def _trap_xy(b, r):
        """Centre of trap block (b, r)."""
        return (b * block_pitch, r * row_pitch)

    def _junc_xy(b, r):
        """Centre of the junction between block b and b+1 in row r."""
        tx, _ = _trap_xy(b, r)
        return (tx + trap_w / 2 + junc_gap / 2, r * row_pitch)

    def _ion_xy(b, r, slot):
        """Position of ion *slot* inside trap (b, r)."""
        tx, ty = _trap_xy(b, r)
        return (tx - trap_inner_w / 2 + slot * ion_sp, ty)

    # ---- 1. Crossing edges (horizontal & vertical) ----
    for r in range(rows):
        for b in range(m - 1):
            tx1, ty1 = _trap_xy(b, r)
            jx, jy = _junc_xy(b, r)
            tx2, ty2 = _trap_xy(b + 1, r)
            re = tx1 + trap_w / 2
            le = tx2 - trap_w / 2
            # Trap → junction
            ax.plot([re, jx - JUNCTION_SIDE / 2], [ty1, jy],
                    color=CROSSING_COLOR, linewidth=CROSSING_LW,
                    solid_capstyle="round", zorder=1)
            # Junction → next trap
            ax.plot([jx + JUNCTION_SIDE / 2, le], [jy, ty2],
                    color=CROSSING_COLOR, linewidth=CROSSING_LW,
                    solid_capstyle="round", zorder=1)
            # Edge label at midpoint
            if show_labels:
                emx = (re + le) / 2
                emy = ty1 + trap_h / 2 + 0.35
                ax.text(emx, emy, f"E({b}\u2194{b+1},r{r})",
                        fontsize=7, ha="center", va="bottom",
                        color="#607D8B", fontstyle="italic",
                        path_effects=_STROKE_THIN)

    for r in range(rows - 1):
        for b in range(m - 1):
            jx1, jy1 = _junc_xy(b, r)
            jx2, jy2 = _junc_xy(b, r + 1)
            ax.plot([jx1, jx2],
                    [jy1 + JUNCTION_SIDE / 2, jy2 - JUNCTION_SIDE / 2],
                    color=CROSSING_VERT_COLOR, linewidth=CROSSING_LW * 0.85,
                    linestyle="--", dash_capstyle="round", zorder=1)

    # ---- 2. Trap blocks ----
    for b in range(m):
        for r in range(rows):
            cx, cy = _trap_xy(b, r)
            rect = FancyBboxPatch(
                (cx - trap_w / 2, cy - trap_h / 2), trap_w, trap_h,
                boxstyle="round,pad=0.20",
                facecolor=TRAP_FILL, edgecolor=TRAP_EDGE,
                linewidth=TRAP_LINEWIDTH, zorder=2, alpha=0.92)
            ax.add_patch(rect)
            ax.text(cx, cy + trap_h / 2 + 0.22,
                    f"T({b},{r}) k={k}",
                    fontsize=TRAP_LABEL_FONT, ha="center", va="bottom",
                    color=TRAP_EDGE, fontweight="bold",
                    path_effects=_STROKE_THIN)

    # ---- 3. Junctions ----
    half = JUNCTION_SIDE / 2
    for b in range(m - 1):
        for r in range(rows):
            jx, jy = _junc_xy(b, r)
            jrect = FancyBboxPatch(
                (jx - half, jy - half), JUNCTION_SIDE, JUNCTION_SIDE,
                boxstyle="round,pad=0.07",
                facecolor=JUNCTION_FILL, edgecolor=JUNCTION_EDGE,
                linewidth=1.5, zorder=4, alpha=0.88)
            ax.add_patch(jrect)
            # Always show junction index
            jlabel = f"J({b},{r})"
            ax.text(jx, jy, jlabel,
                    fontsize=JUNCTION_FONT - 1, ha="center", va="center",
                    color="white", fontweight="bold", zorder=5,
                    path_effects=[path_effects.withStroke(
                        linewidth=2, foreground=JUNCTION_EDGE)])

    # ---- 4. Ions ----
    ion_positions = {}
    if show_ions:
        for r in range(rows):
            for b in range(m):
                for slot in range(k):
                    ion_idx = r * total_cols + b * k + slot
                    ix, iy = _ion_xy(b, r, slot)
                    ion_positions[ion_idx] = (ix, iy)
                    _draw_ion(ax, ix, iy, ion_idx,
                              highlight_set, ion_roles,
                              show_label=show_labels,
                              radius=ION_RADIUS, zorder=6)

    # ---- 5. Highlight routing path ----
    if highlight_qubits and len(highlight_qubits) >= 2:
        hq = [q for q in highlight_qubits if q in ion_positions]
        if len(hq) >= 2:
            xs = [ion_positions[q][0] for q in hq]
            ys = [ion_positions[q][1] for q in hq]
            cx = (min(xs) + max(xs)) / 2
            cy = (min(ys) + max(ys)) / 2
            ew = max(xs) - min(xs) + ION_RADIUS * 6
            eh = max(ys) - min(ys) + ION_RADIUS * 6
            ellip = Ellipse(
                (cx, cy), max(ew, ION_RADIUS * 6), max(eh, ION_RADIUS * 6),
                edgecolor="#FF6600", facecolor=HIGHLIGHT_BG,
                alpha=0.35, linewidth=HIGHLIGHT_LW, linestyle="--", zorder=3)
            ax.add_patch(ellip)
            ax.plot(xs, ys, color="#FF6600", linewidth=2.0,
                    linestyle=":", alpha=0.80, zorder=3,
                    marker="o", markersize=4)

    # ---- 6. Info panel (bottom-left) ----
    info_lines = [
        f"m={m}  n={rows}  k={k}",
        f"Ions: {arch.num_qubits}   Traps: {m*rows}",
        f"Junctions: {(m-1)*rows}",
    ]
    if ion_roles:
        role_counts: Dict[str, int] = {}
        for v in ion_roles.values():
            ch = v[0].upper()
            role_counts[ch] = role_counts.get(ch, 0) + 1
        parts = [f"{rk}:{rv}" for rk, rv in sorted(role_counts.items())]
        info_lines.append("Roles: " + "  ".join(parts))
    info_text = "\n".join(info_lines)
    ax.text(0.01, 0.01, info_text,
            transform=ax.transAxes,
            fontsize=INFO_FONT, va="bottom", ha="left",
            fontfamily="monospace",
            bbox=dict(facecolor="white", alpha=0.88,
                      edgecolor="#bbb", boxstyle="round,pad=0.5"),
            zorder=100)

    # ---- 7. Title & limits ----
    if title:
        ax.set_title(title, fontsize=FONT_SIZE + 2, fontweight="bold", pad=14)
    else:
        ax.set_title(
            f"WISE  m={m} \u00d7 n={rows} \u00d7 k={k}  [{arch.num_qubits} ions]",
            fontsize=FONT_SIZE + 1, fontweight="bold", pad=14)
    ax.set_aspect("equal")
    ax.set_facecolor("#FAFBFE")
    ax.axis("off")

    all_x, all_y = [], []
    for b in range(m):
        for r in range(rows):
            cx, cy = _trap_xy(b, r)
            all_x.extend([cx - trap_w / 2, cx + trap_w / 2])
            all_y.extend([cy - trap_h / 2, cy + trap_h / 2])
    for b in range(m - 1):
        for r in range(rows):
            jx, jy = _junc_xy(b, r)
            all_x.append(jx)
            all_y.append(jy)
    pad = SPACING * 0.9
    if all_x and all_y:
        ax.set_xlim(min(all_x) - pad, max(all_x) + pad)
        ax.set_ylim(min(all_y) - pad - 0.8, max(all_y) + pad + 1.0)


# =============================================================================
# Linear-chain Renderer
# =============================================================================

def _display_linear_chain(arch, fig, ax, title,
                          show_ions, show_labels, highlight_qubits,
                          ion_roles=None):
    """Render a linear-chain architecture with segmented trap zones."""
    n = arch.num_qubits
    sp = SPACING * 0.7
    seg_size = 5
    n_segs = max(1, math.ceil(n / seg_size))
    highlight_set = set(highlight_qubits or [])

    # Trap segments
    for s in range(n_segs):
        i0 = s * seg_size
        i1 = min(i0 + seg_size - 1, n - 1)
        x0 = i0 * sp - sp * 0.4
        x1 = i1 * sp + sp * 0.4
        sw, sh = x1 - x0, sp * 0.65
        is_gate = (s == n_segs // 2)
        face = TRAP_GATE_FILL if is_gate else TRAP_FILL
        edge = TRAP_GATE_EDGE if is_gate else TRAP_EDGE
        rect = FancyBboxPatch(
            (x0, -sh / 2), sw, sh,
            boxstyle="round,pad=0.15",
            facecolor=face, edgecolor=edge,
            alpha=0.88, linewidth=TRAP_LINEWIDTH, zorder=0)
        ax.add_patch(rect)
        label = "gate zone" if is_gate else f"seg {s}"
        ax.text((x0 + x1) / 2, -sh / 2 - sp * 0.22, label,
                fontsize=TRAP_LABEL_FONT, ha="center", va="top",
                color=edge, fontstyle="italic",
                path_effects=_STROKE_THIN)

    # Ion chain backbone line
    if n > 1:
        ax.plot([0, (n - 1) * sp], [0, 0],
                color="#555", linewidth=EDGE_LINEWIDTH, alpha=0.65,
                zorder=1, solid_capstyle="round")

    # Ions
    if show_ions:
        for i in range(n):
            _draw_ion(ax, i * sp, 0, i, highlight_set, ion_roles,
                      show_label=show_labels, radius=ION_RADIUS * 0.85, zorder=5)

    # End-cap markers
    for ex in [0, (n - 1) * sp]:
        ax.plot(ex, 0, marker="|", markersize=16, color="#555", zorder=6)

    if title:
        ax.set_title(title, fontsize=FONT_SIZE + 2, fontweight="bold", pad=10)
    else:
        ax.set_title(f"Linear chain  ({n} ions, {n_segs} segments)",
                     fontsize=FONT_SIZE, fontweight="bold", pad=10)
    ax.set_aspect("equal")
    ax.set_facecolor("#FAFBFE")
    ax.axis("off")
    margin = sp * 1.3
    ax.set_xlim(-margin, (n - 1) * sp + margin)
    ax.set_ylim(-margin, margin)


# =============================================================================
# QCCD Renderer
# =============================================================================

def _display_qccd(arch, fig, ax, title,
                  show_junctions, show_edges, show_ions, show_labels,
                  highlight_qubits, ion_roles=None):
    """Render QCCD architecture with styled traps, junctions, crossings.

    Follows the same visual language as ``_display_wise_grid``:
    FancyBboxPatch traps, ``_draw_ion()`` ion circles with role letters,
    junction boxes, crossing edges, and an info panel.
    """
    graph = arch.qccd_graph
    highlight_set = set(highlight_qubits or [])

    # --- Coordinate helpers --------------------------------------------------
    all_x: List[float] = []
    all_y: List[float] = []

    ion_sp = SPACING * ION_SPACING_RATIO
    trap_pad_x = TRAP_PAD_X
    trap_pad_y = TRAP_PAD_Y

    ion_positions: Dict[int, Tuple[float, float]] = {}

    # ---- 1. Crossing edges (drawn first so traps sit on top) ----------------
    if show_edges:
        seen_pairs: set = set()
        for crossing in graph.crossings.values():
            si = crossing.source.idx
            ti = crossing.target.idx
            pair = (min(si, ti), max(si, ti))
            if pair in seen_pairs:
                continue
            seen_pairs.add(pair)
            s_node = graph.nodes.get(si)
            t_node = graph.nodes.get(ti)
            if s_node is None or t_node is None:
                continue
            sx, sy = s_node.position
            tx, ty = t_node.position
            ax.plot([sx, tx], [sy, ty],
                    color=CROSSING_COLOR, linewidth=CROSSING_LW,
                    solid_capstyle="round", zorder=1, alpha=0.6)
            if show_labels:
                mx, my = (sx + tx) / 2, (sy + ty) / 2
                ax.text(mx, my + 0.6, f"E({si}\u2194{ti})",
                        fontsize=7, ha="center", va="bottom",
                        color="#607D8B", fontstyle="italic",
                        path_effects=_STROKE_THIN)

    # ---- 2. Trap blocks -----------------------------------------------------
    for node in graph.nodes.values():
        _ntype = type(node).__name__
        px, py = node.position

        if "Trap" in _ntype:
            cap = getattr(node, "capacity", 3)
            horiz = getattr(node, "is_horizontal", True)
            node_ions = list(getattr(node, "ions", []))
            ni = len(node_ions)
            node_spacing = getattr(node, "spacing", 1.0)
            if node_spacing <= 0:
                node_spacing = ion_sp

            inner_span = max(0, (ni - 1)) * ion_sp
            if horiz:
                tw = inner_span + 2 * trap_pad_x
                th = 2 * ION_RADIUS + 2 * trap_pad_y
            else:
                tw = 2 * ION_RADIUS + 2 * trap_pad_y
                th = inner_span + 2 * trap_pad_x

            rect = FancyBboxPatch(
                (px - tw / 2, py - th / 2), tw, th,
                boxstyle="round,pad=0.20",
                facecolor=TRAP_FILL, edgecolor=TRAP_EDGE,
                linewidth=TRAP_LINEWIDTH, zorder=2, alpha=0.92)
            ax.add_patch(rect)

            _label = getattr(node, "display_label",
                             getattr(node, "label", f"T{node.idx}"))
            label_y_off = th / 2 + 0.22
            ax.text(px, py + label_y_off,
                    f"{_label} k={cap}",
                    fontsize=TRAP_LABEL_FONT, ha="center", va="bottom",
                    color=TRAP_EDGE, fontweight="bold",
                    path_effects=_STROKE_THIN)

            if show_ions and node_ions:
                for i, ion in enumerate(node_ions):
                    if horiz:
                        ix = px - inner_span / 2 + i * ion_sp
                        iy = py
                    else:
                        ix = px
                        iy = py - inner_span / 2 + i * ion_sp
                    ion_positions[ion.idx] = (ix, iy)
                    _draw_ion(ax, ix, iy, ion.idx,
                              highlight_set, ion_roles,
                              show_label=show_labels,
                              radius=ION_RADIUS, zorder=6)

            all_x.extend([px - tw / 2, px + tw / 2])
            all_y.extend([py - th / 2, py + th / 2])

        elif "Junction" in _ntype or "Crossing" in _ntype:
            if show_junctions:
                half = JUNCTION_SIDE / 2
                jrect = FancyBboxPatch(
                    (px - half, py - half), JUNCTION_SIDE, JUNCTION_SIDE,
                    boxstyle="round,pad=0.07",
                    facecolor=JUNCTION_FILL, edgecolor=JUNCTION_EDGE,
                    linewidth=1.5, zorder=4, alpha=0.88)
                ax.add_patch(jrect)
                _jlabel = getattr(node, "display_label",
                                  getattr(node, "label", f"J{node.idx}"))
                ax.text(px, py, _jlabel,
                        fontsize=JUNCTION_FONT - 1, ha="center", va="center",
                        color="white", fontweight="bold", zorder=5,
                        path_effects=[path_effects.withStroke(
                            linewidth=2, foreground=JUNCTION_EDGE)])
                all_x.append(px)
                all_y.append(py)
        else:
            ax.plot(px, py, "o", color="#888888", markersize=6, zorder=3)
            if show_labels:
                ax.text(px, py + 0.25,
                        getattr(node, "display_label", str(node.idx)),
                        fontsize=6.5, ha="center", va="bottom", color="#555555")
            all_x.append(px)
            all_y.append(py)

    # ---- 3. Highlight routing path ------------------------------------------
    if highlight_qubits and len(highlight_qubits) >= 2:
        hq = [q for q in highlight_qubits if q in ion_positions]
        if len(hq) >= 2:
            xs = [ion_positions[q][0] for q in hq]
            ys = [ion_positions[q][1] for q in hq]
            cx = (min(xs) + max(xs)) / 2
            cy = (min(ys) + max(ys)) / 2
            ew = max(xs) - min(xs) + ION_RADIUS * 6
            eh = max(ys) - min(ys) + ION_RADIUS * 6
            ellip = Ellipse(
                (cx, cy), max(ew, ION_RADIUS * 6), max(eh, ION_RADIUS * 6),
                edgecolor="#FF6600", facecolor=HIGHLIGHT_BG,
                alpha=0.35, linewidth=HIGHLIGHT_LW, linestyle="--", zorder=3)
            ax.add_patch(ellip)
            ax.plot(xs, ys, color="#FF6600", linewidth=2.0,
                    linestyle=":", alpha=0.80, zorder=3,
                    marker="o", markersize=4)

    # ---- 4. Info panel (bottom-left) ----------------------------------------
    n_traps = sum(1 for n in graph.nodes.values() if "Trap" in type(n).__name__)
    n_junctions = sum(1 for n in graph.nodes.values() if "Junction" in type(n).__name__)
    info_lines = [
        f"Traps: {n_traps}   Junctions: {n_junctions}",
        f"Ions: {arch.num_qubits}   k={getattr(arch, 'ions_per_trap', '?')}",
    ]
    if ion_roles:
        role_counts: Dict[str, int] = {}
        for v in ion_roles.values():
            ch = v[0].upper()
            role_counts[ch] = role_counts.get(ch, 0) + 1
        parts = [f"{rk}:{rv}" for rk, rv in sorted(role_counts.items())]
        info_lines.append("Roles: " + "  ".join(parts))
    info_text = "\n".join(info_lines)
    ax.text(0.01, 0.01, info_text,
            transform=ax.transAxes,
            fontsize=INFO_FONT, va="bottom", ha="left",
            fontfamily="monospace",
            bbox=dict(facecolor="white", alpha=0.88,
                      edgecolor="#bbb", boxstyle="round,pad=0.5"),
            zorder=100)

    # ---- 5. Title & limits --------------------------------------------------
    if title:
        ax.set_title(title, fontsize=FONT_SIZE + 2, fontweight="bold", pad=14)
    else:
        ax.set_title(
            f"{type(arch).__name__}  [{arch.num_qubits} ions]",
            fontsize=FONT_SIZE + 1, fontweight="bold", pad=14)
    ax.set_aspect("equal")
    ax.set_facecolor("#FAFBFE")
    ax.axis("off")

    pad = SPACING * 0.9
    if all_x and all_y:
        ax.set_xlim(min(all_x) - pad, max(all_x) + pad)
        ax.set_ylim(min(all_y) - pad - 0.8, max(all_y) + pad + 1.0)


def _display_generic(arch, fig, ax, title):
    """Fallback for unknown architecture types."""
    ax.text(0.5, 0.5, f"{arch.name}\n{arch.num_qubits} qubits",
            transform=ax.transAxes, ha="center", va="center",
            fontsize=20, fontweight="bold")
    if title:
        ax.set_title(title, fontsize=FONT_SIZE + 2, fontweight="bold")
    ax.axis("off")


def _highlight_ions(ax, pos, qubit_indices):
    """Draw a dashed ellipse around highlighted ions in QCCD view."""
    pts = []
    for q in qubit_indices:
        key = f"ion_{q}"
        if key in pos:
            pts.append(pos[key])
    if not pts:
        return
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    cx = (min(xs) + max(xs)) / 2
    cy = (min(ys) + max(ys)) / 2
    w = max(xs) - min(xs) + 2.0
    h = max(ys) - min(ys) + 2.0
    ellip = Ellipse(
        (cx, cy), max(w, 1.5), max(h, 1.5),
        edgecolor="#FF6600", facecolor=HIGHLIGHT_BG,
        alpha=0.35, linewidth=HIGHLIGHT_LW, linestyle="--")
    ax.add_patch(ellip)


# =============================================================================
# Transport Animation
# =============================================================================

def animate_transport(arch, operations, interval=1200, show_labels=True,
                      ion_roles=None, interp_frames=12,
                      gate_hold_frames=18, stim_circuit=None):
    """Animate step-by-step ion transport with actual state simulation.

    Maintains mutable ion positions and updates them as transport
    operations execute, so ions visibly move between trap segments.
    Gate operations highlight the involved ions in-place and draw
    coloured laser beams onto them to distinguish gate types:
      * **MS (two-qubit)** — yellow converging beams
      * **Rotation (single-qubit)** — purple downward beam
      * **Measurement** — green downward beam

    Single-qubit gate operations (rotations, resets) are automatically
    filtered out — they don't produce visible ion movement on WISE
    grids and would otherwise flash as confusing spurious gate frames.

    Transport operations that belong to the same parallel pass (as
    marked by ``PASS_BOUNDARY`` sentinels) are grouped and animated
    as a single atomic step, preventing transient capacity violations.

    Parameters
    ----------
    arch : TrappedIonArchitecture
    operations : list
        Sequence of gate / transport operations.  Transport ops must
        expose ``.qubits`` (tuple of int) and optionally
        ``.source_zone`` / ``.target_zone``.
    interval : int
        Milliseconds between frames (default 1200 for slower pace).
    show_labels : bool
    ion_roles : dict | None
    interp_frames : int
        Number of interpolation sub-frames per transport step.
    gate_hold_frames : int
        Extra hold frames appended to gate steps so the laser-beam
        highlight stays visible.  Total gate frames =
        ``interp_frames + gate_hold_frames``.  Set to 0 to disable.
    stim_circuit : stim.Circuit | None
        If provided, the original stim circuit is displayed as a
        scrolling sidebar alongside the animation with a tracking
        pointer showing the current position.

    Returns
    -------
    matplotlib.animation.FuncAnimation
    """
    _require_matplotlib()
    from matplotlib.animation import FuncAnimation as _FA

    _cls_name = type(arch).__name__
    _mro = _mro_names(arch)
    is_wise = _cls_name == "WISEArchitecture"
    is_qccd = (not is_wise
               and "QCCDArchitecture" in _mro
               and hasattr(arch, "qccd_graph"))

    # ----- Identify single-qubit gate ops (rotations / resets) ----------
    # These come from native-gate decomposition.  Instead of filtering
    # them out entirely, we batch consecutive 1Q gates into one
    # animation frame so users can see rotation "layers" as brief
    # purple flashes without cluttering the animation.

    _QCCD_XPORT_NAMES = frozenset({
        "Split", "Merge", "Move", "JunctionCrossing", "Shuttle",
        "TransportOperation",
    })

    def _is_transport(op) -> bool:
        _cn = type(op).__name__
        if _cn in _QCCD_XPORT_NAMES:
            return True
        return hasattr(op, "source_zone")

    def _is_1q_gate(op):
        if _is_transport(op):  # Transport → never a 1Q gate
            return False
        qs = getattr(op, 'qubits', None) or getattr(op, 'targets', None)
        if qs is not None and len(qs) < 2:
            return True
        return False

    # ----- WISE grid geometry (same constants as _display_wise_grid) -----
    if is_wise:
        rows = arch.rows
        k = arch.ions_per_segment
        m = arch.col_groups
        total_cols = arch.total_columns
    else:
        rows = k = m = total_cols = 0

    ion_sp = SPACING * ION_SPACING_RATIO
    trap_inner_w = max(0, (k - 1) * ion_sp)
    trap_w = trap_inner_w + 2 * TRAP_PAD_X
    trap_h = 2 * ION_RADIUS + 2 * TRAP_PAD_Y
    junc_gap = SPACING * 1.2
    block_pitch = trap_w + junc_gap
    row_pitch = SPACING * 2.5  # tighter than _display_wise_grid for animation

    def _trap_xy(b, r):
        return (b * block_pitch, r * row_pitch)

    def _junc_xy(b, r):
        tx, _ = _trap_xy(b, r)
        return (tx + trap_w / 2 + junc_gap / 2, r * row_pitch)

    def _ion_home(b, r, slot):
        """Default position of ion *slot* inside trap (b, r)."""
        tx, ty = _trap_xy(b, r)
        return (tx - trap_inner_w / 2 + slot * ion_sp, ty)

    # --- Build initial ion → (x, y) map + ion → (block, row) map ---
    ion_pos: Dict[int, Tuple[float, float]] = {}
    ion_trap: Dict[int, Tuple[int, int]] = {}   # ion → (block, row)
    trap_ions: Dict[Tuple[int, int], List[int]] = {}  # (b,r) → [ion_idxs]

    if is_wise:
        for r in range(rows):
            for b in range(m):
                trap_ions[(b, r)] = []
                for slot in range(k):
                    idx = r * total_cols + b * k + slot
                    ion_pos[idx] = _ion_home(b, r, slot)
                    ion_trap[idx] = (b, r)
                    trap_ions[(b, r)].append(idx)

        # Discover any off-grid ion IDs referenced in operations and
        # assign them an initial position so transport/gate ops that
        # reference them don't silently disappear.
        _all_op_ions: set = set()
        for _op in operations:
            for _attr in ("qubits", "targets", "ion_indices"):
                _v = getattr(_op, _attr, None)
                if _v is not None:
                    _all_op_ions.update(int(x) for x in _v if int(x) >= 0)
        _off_grid = sorted(_all_op_ions - set(ion_pos.keys()))
        if _off_grid:
            # Place off-grid ions just below the bottom row, stacked left.
            _staging_y = rows * row_pitch + row_pitch * 0.6
            for _oi, _oion in enumerate(_off_grid):
                ion_pos[_oion] = (_oi * ion_sp, _staging_y)
                # Don't assign a trap — they'll be transported in later

    elif is_qccd:
        # --- QCCD-graph-based ion position init (AugGrid / Networked / etc.) ---
        _qccd_graph = arch.qccd_graph
        _node_positions: Dict[int, Tuple[float, float]] = {}
        _node_ions: Dict[int, List[int]] = {}
        for _node in _qccd_graph.nodes.values():
            _ntype = type(_node).__name__
            _node_positions[_node.idx] = _node.position
            if _ntype in ("ManipulationTrap", "StorageTrap"):
                _node_ions[_node.idx] = []
                _nx, _ny = _node.position
                _ni = len(_node.ions)
                _sp = _node.spacing if _node.spacing > 0 else 0.3
                _horiz = _node.is_horizontal
                for _ii, _ion in enumerate(_node.ions):
                    if _horiz:
                        _ix = _nx - (_ni - 1) * _sp / 2 + _ii * _sp
                        _iy = _ny
                    else:
                        _ix = _nx
                        _iy = _ny - (_ni - 1) * _sp / 2 + _ii * _sp
                    ion_pos[_ion.idx] = (_ix, _iy)
                    ion_trap[_ion.idx] = (_node.idx, 0)
                    _node_ions[_node.idx].append(_ion.idx)
                    trap_ions.setdefault((_node.idx, 0), []).append(_ion.idx)

        # Discover off-grid ions from operations
        _all_op_ions_q: set = set()
        for _op in operations:
            for _attr in ("qubits", "targets", "ion_indices"):
                _v = getattr(_op, _attr, None)
                if _v is not None:
                    _all_op_ions_q.update(int(x) for x in _v if int(x) >= 0)
        _off_grid_q = sorted(_all_op_ions_q - set(ion_pos.keys()))
        if _off_grid_q:
            _all_ys = [p[1] for p in ion_pos.values()] if ion_pos else [0]
            _staging_y_q = max(_all_ys) + 3.0
            for _oi, _oion in enumerate(_off_grid_q):
                ion_pos[_oion] = (_oi * 1.5, _staging_y_q)

    # --- Parse zone string "trap_<row>_<col>" → (block, row, slot) ---
    def _parse_zone(zone_str: str) -> Optional[Tuple[int, int, int]]:
        """Parse zone id to *(block, row, slot)*.

        Zone format is ``trap_<row>_<absolute_col>`` where *absolute_col*
        spans ``0 … total_cols-1``.  We decompose into
        ``block = col // k``, ``slot = col % k`` so that intra-block
        H_SWAPs are resolved to distinct ion slots.

        Returns None on parse failure.
        """
        if zone_str and zone_str.startswith("trap_"):
            parts = zone_str.split("_")
            if len(parts) >= 3:
                try:
                    r_val, c_val = int(parts[1]), int(parts[2])
                    b_val = c_val // k if k > 0 else c_val
                    s_val = c_val % k if k > 0 else 0
                    return (b_val, r_val, s_val)
                except ValueError:
                    pass
        return None

    def _parse_zone_br(zone_str: str) -> Optional[Tuple[int, int]]:
        """Convenience: parse to *(block, row)* only (for trap-level ops)."""
        full = _parse_zone(zone_str)
        return (full[0], full[1]) if full else None

    # --- Classify operations -----------------------------------------------
    def _is_transport(op) -> bool:
        return (type(op).__name__ == "TransportOperation"
                or hasattr(op, "source_zone"))

    def _op_label(op) -> str:
        cls = type(op).__name__
        if _is_transport(op):
            src = getattr(op, "source_zone", "?")
            tgt = getattr(op, "target_zone", "?")
            return f"Transport → {tgt}"
        lbl = getattr(op, "label", None) or getattr(op, "name", None) or cls
        return str(lbl)

    def _op_qubits(op) -> List[int]:
        for attr in ("qubits", "targets", "ion_indices"):
            v = getattr(op, attr, None)
            if v is not None:
                return list(v)
        return []

    # --- Pre-compute per-step snapshots of ion positions ---
    # snapshot[i] = dict of ion_idx → (x, y) AFTER applying op i
    # snapshot[-1] = initial state
    n_ops = len(operations)
    snapshots: List[Dict[int, Tuple[float, float]]] = [dict(ion_pos)]
    active_ions_per_step: List[List[int]] = [[]]
    labels_per_step: List[str] = ["Initial configuration"]
    is_transport_step: List[bool] = [False]

    current_pos = dict(ion_pos)
    current_trap = dict(ion_trap)
    current_trap_ions = {k2: list(v2) for k2, v2 in trap_ions.items()}

    def _recompute_trap_positions(trap_key):
        """Recompute X/Y positions for all ions in *trap_key*."""
        if trap_key not in current_trap_ions:
            return
        members = current_trap_ions[trap_key]
        bt2, rt2 = trap_key
        cx2, cy2 = _trap_xy(bt2, rt2)
        n_in2 = len(members)
        for i3, iid2 in enumerate(members):
            if n_in2 == 1:
                current_pos[iid2] = (cx2, cy2)
            else:
                span2 = min(trap_inner_w, (n_in2 - 1) * ion_sp)
                current_pos[iid2] = (
                    cx2 - span2 / 2 + i3 * (span2 / max(1, n_in2 - 1)),
                    cy2,
                )

    def _apply_transport(ion_idx_t, tgt_parsed):
        """Move *ion_idx_t* into the trap given by *tgt_parsed*.

        *tgt_parsed* is ``(block, row, slot)`` from ``_parse_zone``
        or ``(block, row)`` for legacy callers.
        """
        old_trap_t = current_trap.get(ion_idx_t)
        if old_trap_t and old_trap_t in current_trap_ions:
            if ion_idx_t in current_trap_ions[old_trap_t]:
                current_trap_ions[old_trap_t].remove(ion_idx_t)
        bt3 = tgt_parsed[0]
        rt3 = tgt_parsed[1]
        bt3 = min(bt3, m - 1)
        rt3 = min(rt3, rows - 1)
        clamped = (bt3, rt3)
        if clamped not in current_trap_ions:
            current_trap_ions[clamped] = []
        current_trap_ions[clamped].append(ion_idx_t)
        current_trap[ion_idx_t] = clamped
        return old_trap_t, clamped

    # --- Detect PASS_BOUNDARY sentinel (qubit=-1, zone=__PASS_BOUNDARY__) ---
    def _is_pass_boundary(op) -> bool:
        if _is_transport(op):
            src = getattr(op, "source_zone", None)
            return src == "__PASS_BOUNDARY__"
        return False

    # --- Classify gate type for laser-beam colouring ---
    def _gate_kind(op) -> Optional[str]:
        """Return 'ms', 'rotation', 'measure' or None."""
        # Direct class-based detection (MeasurementOperation/ResetOperation
        # lack .gate_name / .name attributes)
        _cls = type(op).__name__
        if _cls == "MeasurementOperation":
            return "measure"
        if _cls == "ResetOperation":
            return "reset"
        name = (getattr(op, "gate_name", None)
                or getattr(op, "name", None)
                or getattr(op, "label", None)
                or "")
        # Fallback: check op.gate.name (GateOperation wraps GateSpec)
        if not name:
            _gate_obj = getattr(op, "gate", None)
            if _gate_obj is not None:
                name = getattr(_gate_obj, "name", "") or ""
        name_lower = str(name).lower()
        if name_lower in ("ms", "xx", "zz", "cx", "cnot", "cz"):
            return "ms"
        if name_lower in ("m", "mz", "mx", "measure", "measurement"):
            return "measure"
        if name_lower in ("h", "s", "t", "rx", "ry", "rz", "sdg", "tdg",
                          "x", "y", "z", "r", "reset"):
            return "rotation"
        # Check via GateOperation attribute
        op_type = getattr(op, "operation_type", None)
        if op_type is not None:
            ot = str(op_type).lower()
            if "two" in ot or "2q" in ot:
                return "ms"
            if "one" in ot or "1q" in ot:
                return "rotation"
        return None

    # Laser colours
    LASER_MS = "#FFD600"       # yellow for MS/two-qubit
    LASER_ROTATION = "#AB47BC" # purple for rotations
    LASER_MEASURE = "#66BB6A"  # green for measurement
    LASER_RESET = "#00BCD4"    # cyan for reset/initialization

    # Physical timing constants (µs) for proportional animation speed
    try:
        from qectostim.experiments.hardware_simulation.trapped_ion.routing import (
            H_PASS_TIME_US as _H_US,
            V_PASS_TIME_US as _V_US,
        )
    except ImportError:
        _H_US, _V_US = 212.0, 510.0
    _GATE_TIME_US = {"ms": 100.0, "rotation": 10.0,
                     "measure": 100.0, "reset": 5.0}

    # --- Pre-group operations into animation steps ---
    # Transport ops between PASS_BOUNDARY markers form one atomic step.
    # Gate ops are individual steps. Swap-pairs (mirror transports)
    # within a pass group are detected and all applied atomically.

    def _apply_swap_group(swap_list):
        """Apply a list of (ion_a, tgt_a, ion_b, tgt_b) atomically.

        Each *tgt* is ``(block, row, slot)`` from ``_parse_zone``.
        Slot information is used so intra-block swaps place ions
        at the correct column within the trap, instead of blindly
        re-distributing evenly.
        """
        affected_traps = set()
        # Collect explicit slot assignments: (block, row) → {slot: ion}
        slot_assignments: Dict[Tuple[int, int], Dict[int, int]] = {}

        # Phase 1: remove all ions from their current traps
        for q_a_idx, _, q_b_idx, _ in swap_list:
            for qid in (q_a_idx, q_b_idx):
                old_t = current_trap.get(qid)
                if old_t and old_t in current_trap_ions:
                    if qid in current_trap_ions[old_t]:
                        current_trap_ions[old_t].remove(qid)
                    affected_traps.add(old_t)

        # Phase 2: place all ions in their destinations
        for q_a_idx, tgt_a, q_b_idx, tgt_b in swap_list:
            for qid, tgt_full in ((q_a_idx, tgt_a), (q_b_idx, tgt_b)):
                if tgt_full and qid in current_pos:
                    bt4, rt4 = tgt_full[0], tgt_full[1]
                    slot4 = tgt_full[2] if len(tgt_full) >= 3 else None
                    bt4 = min(bt4, m - 1)
                    rt4 = min(rt4, rows - 1)
                    clamped2 = (bt4, rt4)
                    if clamped2 not in current_trap_ions:
                        current_trap_ions[clamped2] = []
                    if len(current_trap_ions[clamped2]) >= k:
                        import warnings
                        warnings.warn(
                            f"Trap {clamped2} capacity {k} exceeded by "
                            f"ion {qid}; routing may be incorrect",
                            stacklevel=2,
                        )
                    current_trap_ions[clamped2].append(qid)
                    current_trap[qid] = clamped2
                    affected_traps.add(clamped2)
                    # Record slot assignment for precise placement
                    if slot4 is not None:
                        slot_assignments.setdefault(clamped2, {})[slot4] = qid

        # Recompute visual positions — use slot info when available
        for tk in affected_traps:
            sa = slot_assignments.get(tk, {})
            if sa and tk in current_trap_ions:
                # Place ions with known slots at exact positions;
                # distribute remaining ions in leftover slots.
                bt5, rt5 = tk
                cx5, cy5 = _trap_xy(bt5, rt5)
                placed_slots: set = set()
                for sl, iid in sa.items():
                    if 0 <= sl < k:
                        x5 = cx5 - trap_inner_w / 2 + sl * ion_sp
                        current_pos[iid] = (x5, cy5)
                        placed_slots.add(sl)
                # Remaining ions get leftover slots in order
                remaining = [i for i in current_trap_ions[tk]
                             if i not in sa.values()]
                free_slots = sorted(s for s in range(k)
                                    if s not in placed_slots)
                for ri, fsl in zip(remaining, free_slots):
                    x5 = cx5 - trap_inner_w / 2 + fsl * ion_sp
                    current_pos[ri] = (x5, cy5)
                # Any overflow ions (shouldn't happen) → re-distribute
                overflow = remaining[len(free_slots):]
                if overflow:
                    _recompute_trap_positions(tk)
            else:
                _recompute_trap_positions(tk)
        return affected_traps

    # Build list of "step groups" — each is a list of operations that
    # form one animation step.
    step_groups: List[List] = []
    i_op = 0
    while i_op < len(operations):
        op = operations[i_op]
        if _is_pass_boundary(op):
            # Skip boundary markers; they just delimit groups
            i_op += 1
            continue
        if _is_transport(op):
            # Collect all transports up to next pass boundary or gate
            group = []
            while i_op < len(operations):
                cur = operations[i_op]
                if _is_pass_boundary(cur):
                    i_op += 1
                    break
                if not _is_transport(cur):
                    break
                group.append(cur)
                i_op += 1
            if group:
                step_groups.append(group)
        else:
            if _is_1q_gate(op):
                # Batch consecutive 1Q gates, then sub-split by gate
                # kind so measurements get their own frame (green beam)
                # distinct from rotations (purple beam).
                batch = [op]
                i_op += 1
                while i_op < len(operations):
                    nxt = operations[i_op]
                    if _is_1q_gate(nxt):
                        batch.append(nxt)
                        i_op += 1
                    else:
                        break
                # Sub-split at gate-kind transitions
                cur_sub = [batch[0]]
                cur_kind = _gate_kind(batch[0])
                for _b_op in batch[1:]:
                    _bk = _gate_kind(_b_op)
                    if _bk != cur_kind:
                        step_groups.append(cur_sub)
                        cur_sub = [_b_op]
                        cur_kind = _bk
                    else:
                        cur_sub.append(_b_op)
                step_groups.append(cur_sub)
            else:
                # Batch consecutive MS (2Q) gates into one parallel step
                _gk_cur = _gate_kind(op)
                if _gk_cur == "ms":
                    ms_batch = [op]
                    i_op += 1
                    while i_op < len(operations):
                        nxt = operations[i_op]
                        if (not _is_transport(nxt)
                                and not _is_1q_gate(nxt)
                                and not _is_pass_boundary(nxt)
                                and _gate_kind(nxt) == "ms"):
                            ms_batch.append(nxt)
                            i_op += 1
                        else:
                            break
                    step_groups.append(ms_batch)
                else:
                    step_groups.append([op])
                    i_op += 1

    # --- Now convert step groups into snapshots ---
    gate_kind_per_step: List[Optional[str]] = [None]   # for initial frame

    for group in step_groups:
        all_xport = all(_is_transport(o) for o in group)
        qubits_in_step = []

        if all_xport and is_wise:
            # Pair up mirror transports into atomic swaps
            swap_list = []
            consumed = set()
            for gi, ga in enumerate(group):
                if gi in consumed:
                    continue
                qa = _op_qubits(ga)
                src_a = getattr(ga, "source_zone", None)
                tgt_a = getattr(ga, "target_zone", None)
                # Look for mirror partner in the group
                partner = None
                for gj in range(gi + 1, len(group)):
                    if gj in consumed:
                        continue
                    gb = group[gj]
                    src_b = getattr(gb, "source_zone", None)
                    tgt_b = getattr(gb, "target_zone", None)
                    if (src_a and tgt_a and src_b and tgt_b
                            and src_a == tgt_b and tgt_a == src_b):
                        partner = gj
                        break
                if partner is not None:
                    consumed.add(partner)
                    gb = group[partner]
                    q_a = qa[0] if qa else -1
                    q_b = _op_qubits(gb)
                    q_b = q_b[0] if q_b else -1
                    tgt_a_parsed = _parse_zone(tgt_a)       # (block, row, slot)
                    tgt_b_parsed = _parse_zone(getattr(gb, "target_zone", None))
                    swap_list.append((q_a, tgt_a_parsed, q_b, tgt_b_parsed))
                    qubits_in_step.extend([q_a, q_b])
                else:
                    # Unpaired transport — apply individually
                    ion_idx = qa[0] if qa else -1
                    tgt_zone = getattr(ga, "target_zone", None)
                    tgt_parsed = _parse_zone(tgt_zone) if tgt_zone else None
                    if tgt_parsed and ion_idx in current_pos:
                        old_t, dest = _apply_transport(ion_idx, tgt_parsed)
                        _recompute_trap_positions(dest)
                        if old_t and old_t != dest:
                            _recompute_trap_positions(old_t)
                    qubits_in_step.append(ion_idx)

            if swap_list:
                _apply_swap_group(swap_list)

            snapshots.append(dict(current_pos))
            active_ions_per_step.append(qubits_in_step)
            labels_per_step.append(
                f"Pass: {len(group)} transports ({len(swap_list)} swaps)")
            is_transport_step.append(True)
            gate_kind_per_step.append(None)
        else:
            # Check if this is a batched group of 1Q gates
            is_1q_batch = (len(group) > 1
                           and all(_is_1q_gate(g) for g in group))
            # Check if this is a batched group of MS (2Q) gates
            is_ms_batch = (len(group) >= 1
                           and all(_gate_kind(g) == "ms" for g in group)
                           and not is_1q_batch)
            if is_ms_batch:
                # Merge all MS gates into one parallel animation frame
                q_all = []
                pair_strs = []
                for g in group:
                    qs = _op_qubits(g)
                    q_all.extend(qs)
                    if len(qs) >= 2:
                        pair_strs.append(f"{qs[0]}\u2194{qs[1]}")
                q_all = list(dict.fromkeys(q_all))
                snapshots.append(dict(current_pos))
                active_ions_per_step.append(q_all)
                pairs_desc = ", ".join(pair_strs[:4])
                if len(pair_strs) > 4:
                    pairs_desc += f" +{len(pair_strs)-4}"
                labels_per_step.append(
                    f"MS Gates \u00d7{len(group)} ({pairs_desc})")
                is_transport_step.append(False)
                gate_kind_per_step.append("ms")
            elif is_1q_batch:
                # Merge all 1Q gates into one frame (same gate kind)
                q_all = []
                for g in group:
                    q_all.extend(_op_qubits(g))
                q_all = list(dict.fromkeys(q_all))  # dedupe, keep order
                snapshots.append(dict(current_pos))
                active_ions_per_step.append(q_all)
                _gk_batch = _gate_kind(group[0]) or "rotation"
                if _gk_batch == "measure":
                    labels_per_step.append(f"Measure \u00d7{len(group)}")
                elif _gk_batch == "reset":
                    labels_per_step.append(f"Reset \u00d7{len(group)}")
                elif _gk_batch == "rotation":
                    labels_per_step.append(f"Rotations \u00d7{len(group)}")
                else:
                    labels_per_step.append(f"Gates \u00d7{len(group)}")
                is_transport_step.append(False)
                gate_kind_per_step.append(_gk_batch)
            else:
                # Gate operation(s) — one snapshot per group entry
                for g_op in group:
                    q = _op_qubits(g_op)
                    snapshots.append(dict(current_pos))
                    active_ions_per_step.append(q)
                    labels_per_step.append(_op_label(g_op))
                    is_transport_step.append(False)
                    gate_kind_per_step.append(_gate_kind(g_op))

    # --- Pre-compute junction waypoints for V-swap (vertical) moves ---
    # Each TransportOperation for a V_SWAP carries metadata with the
    # exact swap column ("swap_col").  We use this to compute the
    # correct junction to route through:  junction at block boundary
    # ``swap_col // k`` (between block b and b+1).
    # Fallback: when metadata is absent, infer block from the ion's
    # pre-step X position.

    # Build a flat map: step_idx → {ion_idx: op} for metadata lookup
    _step_op_map: List[Dict[int, Any]] = [{}]   # step 0 = initial
    for sg in step_groups:
        op_map: Dict[int, Any] = {}
        for _sg_op in sg:
            for _sqid in _op_qubits(_sg_op):
                op_map[_sqid] = _sg_op
        all_xp2 = all(_is_transport(o2) for o2 in sg)
        _is_1q_b = len(sg) > 1 and all(_is_1q_gate(g) for g in sg)
        _is_ms_b = (len(sg) >= 1
                    and all(_gate_kind(g) == "ms" for g in sg)
                    and not _is_1q_b)
        if (all_xp2 and is_wise) or _is_1q_b or _is_ms_b:
            # Batched → 1 snapshot
            _step_op_map.append(op_map)
        else:
            # Individual ops → 1 snapshot per op
            for _ in sg:
                _step_op_map.append(op_map)

    waypoints_per_step: List[Dict[int, List[Tuple[float, float]]]] = [{}]
    for step_i in range(1, len(snapshots)):
        wp: Dict[int, List[Tuple[float, float]]] = {}
        # Get the per-ion op map for this step
        step_ops = _step_op_map[step_i] if step_i < len(_step_op_map) else {}
        for idx_w in active_ions_per_step[step_i]:
            prev_xy = snapshots[step_i - 1].get(idx_w)
            next_xy = snapshots[step_i].get(idx_w)
            if prev_xy and next_xy:
                px_prev, py_prev = prev_xy
                _, py_next = next_xy
                if abs(py_prev - py_next) > 0.05:
                    # --- Determine the junction block from metadata ---
                    _ion_op = step_ops.get(idx_w)
                    _meta = getattr(_ion_op, "metadata", None) or {}
                    _swap_col = _meta.get("swap_col")
                    if _swap_col is not None and k > 0:
                        # Junction lives at block boundary swap_col//k
                        # (between block b and block b+1).
                        # For a V_SWAP at absolute col c, the ion
                        # travels through the junction to the right of
                        # block c//k if c is the rightmost slot, or
                        # the junction to the left otherwise.
                        _sw_block = _swap_col // k
                        _sw_slot = _swap_col % k
                        # Right junction of block b has index b
                        # Left junction of block b has index b-1
                        if _sw_slot >= k // 2 and _sw_block <= m - 2:
                            junc_b = _sw_block
                        elif _sw_block - 1 >= 0:
                            junc_b = _sw_block - 1
                        elif _sw_block <= m - 2:
                            junc_b = _sw_block
                        else:
                            junc_b = max(0, m - 2)
                        jx, _ = _junc_xy(junc_b, 0)
                        # 2-waypoint path: horizontal to junction,
                        # then vertical through junction
                        wp[idx_w] = [(jx, py_prev), (jx, py_next)]
                    else:
                        # Fallback: infer block from X position
                        ion_block = max(0, min(
                            m - 1,
                            round(px_prev / block_pitch)
                            if block_pitch > 0 else 0
                        ))
                        candidates = []
                        if ion_block <= m - 2:
                            candidates.append(ion_block)
                        if ion_block - 1 >= 0:
                            candidates.append(ion_block - 1)
                        if not candidates and m >= 2:
                            candidates.append(0)
                        best_junc = None
                        best_dist = float("inf")
                        for bj in candidates:
                            jx2, _ = _junc_xy(bj, 0)
                            dist = abs(px_prev - jx2)
                            if dist < best_dist:
                                best_dist = dist
                                best_junc = [(jx2, py_prev), (jx2, py_next)]
                        if best_junc:
                            wp[idx_w] = best_junc
        waypoints_per_step.append(wp)

    # ----- Figure setup ---------------------------------------------------
    n_ops = len(snapshots) - 1
    interp_frames = max(1, interp_frames)
    gate_hold_frames = max(0, gate_hold_frames)

    # Build per-step frame counts proportional to physical µs.
    # Transport: H-pass ~212µs, V-pass ~510µs.
    # Gates: MS ~100µs, rotation ~10µs, measure ~100µs, reset ~5µs.
    _US_PER_FRAME = max(1.0, _H_US / max(1, interp_frames))  # ~17.7 µs/frame
    _step_nframes: List[int] = []
    _step_duration_us: List[float] = []  # physical duration per step
    for _si in range(n_ops):
        if is_transport_step[_si + 1]:
            # Detect V-swap from metadata
            _has_v = False
            _si_ops = _step_op_map[_si + 1] if _si + 1 < len(_step_op_map) else {}
            for _sop in _si_ops.values():
                _smeta = getattr(_sop, "metadata", None) or {}
                if _smeta.get("swap_type") == "V_SWAP":
                    _has_v = True
                    break
            _t_us = _V_US if _has_v else _H_US
            _nf = max(6, int(_t_us / _US_PER_FRAME))
            _step_nframes.append(_nf)
            _step_duration_us.append(_t_us)
        else:
            _gk_si = gate_kind_per_step[_si + 1] or "rotation"
            _g_us = _GATE_TIME_US.get(_gk_si, 10.0)
            _nf = max(8, int(_g_us / _US_PER_FRAME) + gate_hold_frames)
            _step_nframes.append(_nf)
            _step_duration_us.append(_g_us)

    # Cumulative frame offsets: _cum[i] = first frame of step i.
    # Frame 0 is the initial-state frame (before any steps).
    _cum: List[int] = [1]  # step 0 starts at frame 1
    for _nf in _step_nframes:
        _cum.append(_cum[-1] + _nf)
    total_frames = _cum[-1] if _step_nframes else 1

    if is_wise:
        fw = max(9, m * 4.5)
        fh = max(6, rows * 3.8)
    elif is_qccd:
        _all_xs_q = [p[0] for p in ion_pos.values()] if ion_pos else [0, 10]
        _all_ys_q = [p[1] for p in ion_pos.values()] if ion_pos else [0, 10]
        fw = max(12, (max(_all_xs_q) - min(_all_xs_q)) * 0.5 + 4)
        fh = max(9, (max(_all_ys_q) - min(_all_ys_q)) * 0.5 + 4)
    else:
        fw, fh = 12, 9

    # --- Parse stim circuit for sidebar display ---
    _stim_lines: List[str] = []
    _stim_gate_line_idxs: List[int] = []   # indices of gate lines
    _gate_step_indices: List[int] = []     # animation steps that are gates
    ax_stim = None
    if stim_circuit is not None:
        _stim_str = str(stim_circuit)
        _stim_lines = _stim_str.strip().split('\n')
        # Identify "gate" lines (not TICK, annotations, coords, braces)
        _skip_prefixes = ('TICK', 'DETECTOR', 'OBSERVABLE_INCLUDE',
                          'QUBIT_COORDS', '#', 'REPEAT', '{', '}')
        for _li, _line in enumerate(_stim_lines):
            _stripped = _line.strip()
            if _stripped and not _stripped.startswith(_skip_prefixes):
                _stim_gate_line_idxs.append(_li)
        # Identify animation steps that are gate operations
        for _gi in range(1, len(gate_kind_per_step)):
            if gate_kind_per_step[_gi] is not None:
                _gate_step_indices.append(_gi)

    if stim_circuit is not None and _stim_lines:
        import matplotlib.gridspec as _gs
        fig = plt.figure(figsize=(fw + 5, fh + 1.2), dpi=min(DPI, 120))
        _spec = _gs.GridSpec(1, 2, width_ratios=[3, 1], wspace=0.05)
        ax = fig.add_subplot(_spec[0])
        ax_stim = fig.add_subplot(_spec[1])
    else:
        fig, ax = plt.subplots(figsize=(fw, fh + 1.2), dpi=min(DPI, 120))

    # Pre-compute axis limits
    all_x, all_y = [], []
    if is_wise:
        for b in range(m):
            for r in range(rows):
                cx, cy = _trap_xy(b, r)
                all_x.extend([cx - trap_w / 2, cx + trap_w / 2])
                all_y.extend([cy - trap_h / 2, cy + trap_h / 2])
        for b in range(m - 1):
            for r in range(rows):
                jx, jy = _junc_xy(b, r)
                all_x.append(jx)
                all_y.append(jy)
    elif is_qccd:
        for _nod in arch.qccd_graph.nodes.values():
            _px, _py = _nod.position
            all_x.append(_px)
            all_y.append(_py)
        for _pxy in ion_pos.values():
            all_x.append(_pxy[0])
            all_y.append(_pxy[1])
    pad = SPACING * 0.8
    x_lo = min(all_x, default=0) - pad
    x_hi = max(all_x, default=10) + pad + 1.5  # extra for right-side labels
    y_lo = min(all_y, default=0) - pad - 0.8
    y_hi = max(all_y, default=10) + pad + 1.2

    def _ease(t):
        return t * t * (3 - 2 * t)

    def _draw_grid(ax_local):
        """Draw the static WISE infrastructure (traps, junctions, edges)."""
        # Crossing edges
        for r in range(rows):
            for b in range(m - 1):
                tx1, ty1 = _trap_xy(b, r)
                jx, jy = _junc_xy(b, r)
                tx2, ty2 = _trap_xy(b + 1, r)
                re = tx1 + trap_w / 2
                le = tx2 - trap_w / 2
                ax_local.plot([re, jx - JUNCTION_SIDE / 2], [ty1, jy],
                              color=CROSSING_COLOR, linewidth=1.5,
                              solid_capstyle="round", zorder=1)
                ax_local.plot([jx + JUNCTION_SIDE / 2, le], [jy, ty2],
                              color=CROSSING_COLOR, linewidth=1.5,
                              solid_capstyle="round", zorder=1)
        for r in range(rows - 1):
            for b in range(m - 1):
                jx1, jy1 = _junc_xy(b, r)
                jx2, jy2 = _junc_xy(b, r + 1)
                ax_local.plot(
                    [jx1, jx2],
                    [jy1 + JUNCTION_SIDE / 2, jy2 - JUNCTION_SIDE / 2],
                    color=CROSSING_VERT_COLOR, linewidth=1.2,
                    linestyle="--", zorder=1)

        # Trap rectangles
        for b in range(m):
            for r in range(rows):
                cx, cy = _trap_xy(b, r)
                rect = FancyBboxPatch(
                    (cx - trap_w / 2, cy - trap_h / 2), trap_w, trap_h,
                    boxstyle="round,pad=0.18",
                    facecolor=TRAP_FILL, edgecolor=TRAP_EDGE,
                    linewidth=1.8, zorder=2, alpha=0.88)
                ax_local.add_patch(rect)
                ax_local.text(cx, cy + trap_h / 2 + 0.15,
                              f"T({b},{r})",
                              fontsize=8, ha="center", va="bottom",
                              color=TRAP_EDGE, fontweight="bold",
                              path_effects=_STROKE_THIN)

        # Junctions
        half = JUNCTION_SIDE / 2
        for b in range(m - 1):
            for r in range(rows):
                jx, jy = _junc_xy(b, r)
                jrect = FancyBboxPatch(
                    (jx - half, jy - half), JUNCTION_SIDE, JUNCTION_SIDE,
                    boxstyle="round,pad=0.06",
                    facecolor=JUNCTION_FILL, edgecolor=JUNCTION_EDGE,
                    linewidth=1.2, zorder=4, alpha=0.85)
                ax_local.add_patch(jrect)

    def _draw_grid_qccd(ax_local):
        """Draw the static QCCD-graph-based infrastructure (traps, junctions, edges).

        Works for *any* QCCDArchitecture subclass that exposes ``qccd_graph``.
        """
        _qg = arch.qccd_graph

        # --- Edges (draw first so traps/junctions sit on top) ---
        for _edge in getattr(_qg, 'edges', getattr(_qg, '_edges', [])):
            _src_idx = getattr(_edge, 'src', getattr(_edge, 'source', None))
            _tgt_idx = getattr(_edge, 'tgt', getattr(_edge, 'target', None))
            if _src_idx is None or _tgt_idx is None:
                continue
            _src_node = _qg.nodes.get(_src_idx if isinstance(_src_idx, int) else _src_idx.idx)
            _tgt_node = _qg.nodes.get(_tgt_idx if isinstance(_tgt_idx, int) else _tgt_idx.idx)
            if _src_node is None or _tgt_node is None:
                continue
            sx, sy = _src_node.position
            tx, ty = _tgt_node.position
            ax_local.plot([sx, tx], [sy, ty],
                          color=CROSSING_COLOR, linewidth=1.4,
                          solid_capstyle="round", zorder=1, alpha=0.6)

        # --- Nodes (traps and junctions) ---
        for _node in _qg.nodes.values():
            _ntype = type(_node).__name__
            px, py = _node.position
            _label = getattr(_node, 'display_label',
                             getattr(_node, 'label', str(_node.idx)))

            if 'Trap' in _ntype:
                _tw = 1.6
                _th = 0.7
                rect = FancyBboxPatch(
                    (px - _tw / 2, py - _th / 2), _tw, _th,
                    boxstyle="round,pad=0.15",
                    facecolor=TRAP_FILL, edgecolor=TRAP_EDGE,
                    linewidth=1.6, zorder=2, alpha=0.88)
                ax_local.add_patch(rect)
                ax_local.text(px, py + _th / 2 + 0.12, _label,
                              fontsize=7.5, ha="center", va="bottom",
                              color=TRAP_EDGE, fontweight="bold",
                              path_effects=_STROKE_THIN)
            elif 'Junction' in _ntype or 'Crossing' in _ntype:
                _js = 0.35
                jrect = FancyBboxPatch(
                    (px - _js, py - _js), _js * 2, _js * 2,
                    boxstyle="round,pad=0.06",
                    facecolor=JUNCTION_FILL, edgecolor=JUNCTION_EDGE,
                    linewidth=1.2, zorder=4, alpha=0.85)
                ax_local.add_patch(jrect)
            else:
                # Generic node — small circle
                ax_local.plot(px, py, 'o', color='#888888',
                              markersize=6, zorder=3)
                ax_local.text(px, py + 0.25, _label,
                              fontsize=6.5, ha="center", va="bottom",
                              color='#555555')

    def _draw_ions(ax_local, positions, active_set, show_trail=None,
                   gate_kind=None):
        """Draw all ions at their current positions.

        Parameters
        ----------
        gate_kind : str or None
            If this step is a gate, one of 'ms', 'rotation', 'measure'.
            Draws laser beams on active ions accordingly.
        """
        ion_r = ION_RADIUS * 0.85
        laser_offset = trap_h * 0.7  # beam starts above the ion

        for idx, (ix, iy) in sorted(positions.items()):
            is_active = idx in active_set
            if is_active:
                color = HIGHLIGHT_ION
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

            circ = Circle((ix, iy), ion_r,
                          facecolor=color, edgecolor=edge_c,
                          linewidth=lw, zorder=10)
            ax_local.add_patch(circ)
            role_ch = _ion_role_letter(idx, ion_roles)
            ax_local.text(ix, iy, role_ch,
                          fontsize=9, ha="center", va="center",
                          fontweight="bold", color="white", zorder=11,
                          path_effects=[path_effects.withStroke(
                              linewidth=2, foreground=color)])
            if show_labels:
                ax_local.text(ix, iy - ion_r - 0.12, str(idx),
                              fontsize=7, ha="center", va="top",
                              fontweight="bold", color="#555", zorder=11)

        # --- Laser beams for gate steps ---
        if gate_kind and active_set:
            if gate_kind == "ms":
                beam_color = LASER_MS
            elif gate_kind == "rotation":
                beam_color = LASER_ROTATION
            elif gate_kind == "measure":
                beam_color = LASER_MEASURE
            else:
                beam_color = None

            if beam_color:
                active_positions = [
                    (idx, positions[idx])
                    for idx in active_set
                    if idx in positions
                ]
                if gate_kind == "ms" and len(active_positions) >= 2:
                    # Two converging beams from above meeting at midpoint
                    xs = [p[0] for _, p in active_positions]
                    ys = [p[1] for _, p in active_positions]
                    mid_x = sum(xs) / len(xs)
                    mid_y = sum(ys) / len(ys)
                    for _, (ix, iy) in active_positions:
                        # Beam from above, converging to the ion
                        bx_start = ix
                        by_start = iy + laser_offset
                        ax_local.plot(
                            [bx_start, ix], [by_start, iy + ion_r],
                            color=beam_color, linewidth=6.0, alpha=0.85,
                            solid_capstyle="round", zorder=8)
                        # Large glow circle at ion
                        glow = Circle((ix, iy), ion_r * 2.8,
                                      facecolor=beam_color, alpha=0.40,
                                      edgecolor="none", zorder=9)
                        ax_local.add_patch(glow)
                        # Outer halo
                        halo = Circle((ix, iy), ion_r * 4.0,
                                      facecolor=beam_color, alpha=0.12,
                                      edgecolor="none", zorder=7)
                        ax_local.add_patch(halo)
                    # Thick connecting beam between the two ions
                    if len(active_positions) == 2:
                        (_, (x1, y1)), (_, (x2, y2)) = active_positions
                        ax_local.plot(
                            [x1, x2], [y1, y2],
                            color=beam_color, linewidth=5.0, alpha=0.70,
                            solid_capstyle="round", zorder=8)
                        # Glow line behind it
                        ax_local.plot(
                            [x1, x2], [y1, y2],
                            color=beam_color, linewidth=12.0, alpha=0.18,
                            solid_capstyle="round", zorder=6)
                    # Coloured ring outlines on active ions
                    for _, (ix, iy) in active_positions:
                        ring = Circle((ix, iy), ion_r * 1.1,
                                      facecolor="none",
                                      edgecolor=beam_color,
                                      linewidth=2.5, zorder=11)
                        ax_local.add_patch(ring)
                    # Gate-type label above the pair
                    _ms_x = sum(p[0] for _, p in active_positions) / len(active_positions)
                    _ms_y = max(p[1] for _, p in active_positions) + laser_offset * 1.2
                    ax_local.text(
                        _ms_x, _ms_y, "MS",
                        fontsize=12, ha="center", va="bottom",
                        fontweight="bold", color=beam_color,
                        zorder=12,
                        path_effects=[path_effects.withStroke(
                            linewidth=3, foreground="white")])
                else:
                    # Single downward beam per ion (rotation / measurement)
                    for _, (ix, iy) in active_positions:
                        bx_start = ix
                        by_start = iy + laser_offset * 1.5
                        # Thick visible beam
                        ax_local.plot(
                            [bx_start, ix], [by_start, iy + ion_r],
                            color=beam_color, linewidth=5.0, alpha=0.85,
                            solid_capstyle="round", zorder=8)
                        # Glow behind beam
                        ax_local.plot(
                            [bx_start, ix], [by_start, iy + ion_r],
                            color=beam_color, linewidth=12.0, alpha=0.15,
                            solid_capstyle="round", zorder=7)
                        # Glow circle at ion
                        glow = Circle((ix, iy), ion_r * 2.2,
                                      facecolor=beam_color, alpha=0.35,
                                      edgecolor="none", zorder=9)
                        ax_local.add_patch(glow)
                        # Outer halo
                        halo = Circle((ix, iy), ion_r * 3.5,
                                      facecolor=beam_color, alpha=0.10,
                                      edgecolor="none", zorder=7)
                        ax_local.add_patch(halo)
                        # Coloured ring outline on active ion
                        ring = Circle((ix, iy), ion_r * 1.1,
                                      facecolor="none",
                                      edgecolor=beam_color,
                                      linewidth=2.5, zorder=11)
                        ax_local.add_patch(ring)
                    # Gate-type label above the group
                    if active_positions:
                        _lbl = {"rotation": "ROT", "measure": "MEAS"}.get(
                            gate_kind, "GATE")
                        _lx = sum(p[0] for _, p in active_positions) / len(active_positions)
                        _ly = max(p[1] for _, p in active_positions) + laser_offset * 1.7
                        ax_local.text(
                            _lx, _ly, _lbl,
                            fontsize=11, ha="center", va="bottom",
                            fontweight="bold", color=beam_color,
                            zorder=12,
                            path_effects=[path_effects.withStroke(
                                linewidth=3, foreground="white")])

        # Draw arrows for active transport ions showing motion
        if show_trail:
            for idx_t, (old_xy, new_xy) in show_trail.items():
                if old_xy != new_xy:
                    ax_local.annotate(
                        "", xy=new_xy, xytext=old_xy,
                        arrowprops=dict(arrowstyle="-|>",
                                        color="#E65100", lw=2.0,
                                        alpha=0.6,
                                        connectionstyle="arc3,rad=0.15"),
                        zorder=9)

    def _update(frame):
        ax.clear()
        ax.set_facecolor("#FAFBFE")
        ax.axis("off")
        ax.set_xlim(x_lo, x_hi)
        ax.set_ylim(y_lo, y_hi)
        ax.set_aspect("equal")

        if not is_wise and not is_qccd:
            ax.text(0.5, 0.5, "Animation not available for this architecture",
                    transform=ax.transAxes, ha="center", fontsize=14)
            return

        if is_wise:
            _draw_grid(ax)
        else:
            _draw_grid_qccd(ax)

        if frame == 0:
            # Initial state
            _draw_ions(ax, snapshots[0], set())
            step_label = labels_per_step[0]
            op_detail = ""
            title_str = "Initial configuration"
        else:
            # Variable-length step lookup
            import bisect as _bisect_mod
            op_idx = _bisect_mod.bisect_right(_cum, frame) - 1
            op_idx = max(0, min(op_idx, n_ops - 1))
            _sf = _cum[op_idx]          # first frame of this step
            _nf = _step_nframes[op_idx]  # total frames for this step
            sub = frame - _sf            # sub-frame within this step
            # Interpolation parameter: ramp during interp_frames,
            # clamp at 1.0 during any hold frames.
            if sub < interp_frames:
                t = _ease((sub + 1) / interp_frames)
            else:
                t = 1.0  # hold phase — gate highlight stays visible

            prev_snap = snapshots[op_idx]
            next_snap = snapshots[op_idx + 1]
            active = set(active_ions_per_step[op_idx + 1])
            step_wp = waypoints_per_step[op_idx + 1]

            # Interpolate positions — with junction waypoints for V-swaps
            interp_pos = {}
            trails = {}
            for idx2 in set(prev_snap) | set(next_snap):
                px, py = prev_snap.get(idx2, next_snap.get(idx2, (0, 0)))
                nx2, ny2 = next_snap.get(idx2, prev_snap.get(idx2, (0, 0)))

                if idx2 in step_wp:
                    # Multi-segment path through junction waypoints
                    # (horizontal exit → vertical through junction → horizontal entry)
                    _wps = step_wp[idx2]
                    _pts = [(px, py)] + list(_wps) + [(nx2, ny2)]
                    _n_seg = len(_pts) - 1
                    _seg_lens = []
                    for _si2 in range(_n_seg):
                        _dx = _pts[_si2+1][0] - _pts[_si2][0]
                        _dy = _pts[_si2+1][1] - _pts[_si2][1]
                        _seg_lens.append(math.sqrt(_dx*_dx + _dy*_dy))
                    _tot_len = sum(_seg_lens) or 1.0
                    _cum_t = 0.0
                    ix2, iy2 = px, py  # fallback
                    for _si2 in range(_n_seg):
                        _seg_frac = _seg_lens[_si2] / _tot_len
                        if _cum_t + _seg_frac >= t - 1e-9 or _si2 == _n_seg - 1:
                            _lt = ((t - _cum_t) / _seg_frac
                                   if _seg_frac > 0 else 1.0)
                            _lt = min(1.0, max(0.0, _lt))
                            _x0, _y0 = _pts[_si2]
                            _x1, _y1 = _pts[_si2 + 1]
                            ix2 = _x0 + (_x1 - _x0) * _lt
                            iy2 = _y0 + (_y1 - _y0) * _lt
                            break
                        _cum_t += _seg_frac
                else:
                    # Straight-line interpolation (H-swap or gate)
                    ix2 = px + (nx2 - px) * t
                    iy2 = py + (ny2 - py) * t

                interp_pos[idx2] = (ix2, iy2)
                if idx2 in active and (abs(px - nx2) > 0.1 or abs(py - ny2) > 0.1):
                    trails[idx2] = ((px, py), (ix2, iy2))

            gk = gate_kind_per_step[op_idx + 1]
            # Background tint during gate steps
            if gk == "ms":
                from matplotlib.patches import FancyBboxPatch
                bg_rect = FancyBboxPatch(
                    (x_lo, y_lo), x_hi - x_lo, y_hi - y_lo,
                    boxstyle="round,pad=0",
                    facecolor="#FFD600", alpha=0.07,
                    edgecolor="none", zorder=0)
                ax.add_patch(bg_rect)
            elif gk == "measure":
                from matplotlib.patches import FancyBboxPatch
                bg_rect = FancyBboxPatch(
                    (x_lo, y_lo), x_hi - x_lo, y_hi - y_lo,
                    boxstyle="round,pad=0",
                    facecolor="#66BB6A", alpha=0.05,
                    edgecolor="none", zorder=0)
                ax.add_patch(bg_rect)
            elif gk == "rotation":
                from matplotlib.patches import FancyBboxPatch
                bg_rect = FancyBboxPatch(
                    (x_lo, y_lo), x_hi - x_lo, y_hi - y_lo,
                    boxstyle="round,pad=0",
                    facecolor="#AB47BC", alpha=0.04,
                    edgecolor="none", zorder=0)
                ax.add_patch(bg_rect)

            _draw_ions(ax, interp_pos, active, show_trail=trails,
                       gate_kind=gk)
            step_label = f"Step {op_idx+1}/{n_ops}"
            op_detail = labels_per_step[op_idx + 1]
            ions_str = ",".join(str(q) for q in active_ions_per_step[op_idx + 1])
            is_xp = is_transport_step[op_idx + 1]
            if is_xp:
                kind = "[MOVE]"
            elif gk == "ms":
                kind = "[MS GATE]"
            elif gk == "rotation":
                kind = "[ROTATION]"
            elif gk == "measure":
                kind = "[MEASURE]"
            else:
                kind = "[GATE]"
            title_str = f"{kind} {op_detail}"

        ax.set_title(title_str, fontsize=11, fontweight="bold", pad=8)

        # Status panel
        ax.text(0.02, 0.02, step_label,
                transform=ax.transAxes,
                fontsize=10, va="bottom", ha="left", fontweight="bold",
                bbox=dict(facecolor="white", alpha=0.92,
                          edgecolor="#bbb", boxstyle="round,pad=0.3"),
                zorder=100)
        if frame > 0 and op_detail:
            qb_str = ",".join(str(q) for q in active_ions_per_step[
                min(op_idx + 1, n_ops)])
            ax.text(0.02, 0.09, f"ions=[{qb_str}]  {op_detail}",
                    transform=ax.transAxes,
                    fontsize=8, va="bottom", ha="left", color="#333",
                    bbox=dict(facecolor="#FFF9C4", alpha=0.88,
                              edgecolor="#E0C050", boxstyle="round,pad=0.25"),
                    zorder=100)

        # Gate-type legend (right side)
        legend_y = 0.02
        for lbl, lc in [("MS/2Q", LASER_MS),
                         ("Rotation", LASER_ROTATION),
                         ("Measure", LASER_MEASURE)]:
            ax.plot([], [], color=lc, linewidth=3, label=lbl)
        ax.legend(loc="lower right", fontsize=8, framealpha=0.85,
                  edgecolor="#ccc", fancybox=True, ncol=3,
                  bbox_to_anchor=(0.98, 0.01))

        # --- Stim circuit sidebar rendering ---
        if ax_stim is not None and _stim_lines:
            ax_stim.clear()
            ax_stim.set_facecolor('#F8F8FC')
            ax_stim.axis('off')
            ax_stim.set_xlim(0, 1)
            ax_stim.set_ylim(0, 1)
            ax_stim.set_title('Stim Circuit', fontsize=9,
                              fontweight='bold', pad=4)

            # Determine which stim line to highlight
            _cur_stim_line = 0
            if frame > 0:
                _n_gates_done = sum(
                    1 for _gsi in _gate_step_indices
                    if _gsi <= op_idx + 1
                )
                _n_stim_gates = len(_stim_gate_line_idxs)
                if _n_stim_gates > 0 and _n_gates_done > 0:
                    _prog = min(1.0, _n_gates_done / max(1, len(_gate_step_indices)))
                    _sg_idx2 = min(
                        _n_stim_gates - 1,
                        int(_prog * _n_stim_gates))
                    _cur_stim_line = _stim_gate_line_idxs[_sg_idx2]

            # Show a scrolling window centred on current line
            _window = min(25, len(_stim_lines))
            _w_start = max(0, _cur_stim_line - _window // 2)
            _w_end = min(len(_stim_lines), _w_start + _window)
            if _w_end - _w_start < _window:
                _w_start = max(0, _w_end - _window)

            for _vi, _lidx in enumerate(range(_w_start, _w_end)):
                _is_cur = (_lidx == _cur_stim_line and frame > 0)
                _y_frac = 1.0 - (_vi + 0.5) / _window
                # Highlight background for current line
                if _is_cur:
                    ax_stim.axhspan(
                        _y_frac - 0.5 / _window,
                        _y_frac + 0.5 / _window,
                        facecolor='#FFD600', alpha=0.35, zorder=0)
                    # Pointer arrow
                    ax_stim.text(
                        0.98, _y_frac, '\u25C0',
                        transform=ax_stim.transAxes,
                        fontsize=10, color='#E65100',
                        fontweight='bold', ha='right', va='center',
                        zorder=5)
                _line_txt = _stim_lines[_lidx]
                _trunc = _line_txt[:42]
                ax_stim.text(
                    0.02, _y_frac,
                    f'{_lidx+1:3d}\u2502 {_trunc}',
                    transform=ax_stim.transAxes,
                    fontsize=5.5, fontfamily='monospace',
                    color='#000' if _is_cur else '#888',
                    fontweight='bold' if _is_cur else 'normal',
                    va='center', zorder=3)

    anim = _FA(fig, _update, frames=total_frames,
               interval=interval,
               repeat=False)
    return anim


# =============================================================================
# WISE Reconfiguration
# =============================================================================

def visualize_reconfiguration(arch, phases, fig=None,
                              title="WISE Reconfiguration", ion_roles=None):
    """Render phases of a WISE reconfiguration as a grid of sub-plots.

    Parameters
    ----------
    arch : WISEArchitecture
    phases : list[dict[int, tuple]]
        Each phase maps ``ion_idx -> (x, y)`` positions.
    fig : Figure | None
    title : str
    ion_roles : dict | None

    Returns
    -------
    Figure
    """
    _require_matplotlib()
    n_phases = len(phases)
    if n_phases == 0:
        fig, ax = plt.subplots(dpi=DPI)
        ax.text(0.5, 0.5, "No reconfiguration phases",
                ha="center", va="center", fontsize=FONT_SIZE)
        return fig

    cols = min(n_phases, 4)
    nrows = max(1, math.ceil(n_phases / cols))
    fig, axes = plt.subplots(nrows, cols, figsize=(5.5 * cols, 4.5 * nrows),
                             dpi=DPI, squeeze=False)
    axes_flat = axes.flat

    for idx, phase in enumerate(phases):
        axp = axes_flat[idx]
        axp.set_title(f"Phase {idx}", fontsize=FONT_SIZE - 1, fontweight="bold")
        for ion_idx, (ix, iy) in phase.items():
            _draw_ion(axp, ix, iy, ion_idx, set(), ion_roles,
                      show_label=True, radius=ION_RADIUS * 0.75, zorder=5)
        if idx > 0:
            prev = phases[idx - 1]
            for ion_idx, (ix, iy) in phase.items():
                if ion_idx in prev:
                    px, py = prev[ion_idx]
                    if abs(px - ix) > 0.01 or abs(py - iy) > 0.01:
                        axp.annotate("", xy=(ix, iy), xytext=(px, py),
                                     arrowprops=dict(arrowstyle="-|>",
                                                     color=ANCILLA_COLOR,
                                                     lw=1.8, alpha=0.6,
                                                     connectionstyle="arc3,rad=0.12"))
        axp.set_aspect("equal")
        axp.axis("off")

    for idx in range(n_phases, nrows * cols):
        axes_flat[idx].axis("off")

    fig.suptitle(title, fontsize=FONT_SIZE + 2, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    return fig


# =============================================================================
# Helper: visualize routing result
# =============================================================================

def visualize_routing_result(arch, routing_result, ion_roles=None,
                             title="Routing Result"):
    """Quick visualisation of a RoutedCircuit.

    Draws the architecture with highlighted ion pairs that were routed,
    plus a summary of routing operations.

    Parameters
    ----------
    arch : WISEArchitecture
    routing_result : RoutedCircuit
    ion_roles : dict | None
    title : str

    Returns
    -------
    (Figure, Axes)
    """
    _require_matplotlib()
    ops = routing_result.operations if hasattr(routing_result, 'operations') else []
    involved = set()
    for op in ops:
        if hasattr(op, 'qubits'):
            for q in op.qubits:
                involved.add(q)
    fig, ax = display_architecture(
        arch, title=title,
        highlight_qubits=list(involved),
        ion_roles=ion_roles,
    )
    overhead = getattr(routing_result, 'routing_overhead', 0)
    ax.text(0.99, 0.01,
            f"Routing ops: {overhead}\nInvolved ions: {len(involved)}",
            transform=ax.transAxes,
            fontsize=INFO_FONT, va="bottom", ha="right",
            fontfamily="monospace",
            bbox=dict(facecolor="#E8F5E9", alpha=0.9,
                      edgecolor="#66BB6A", boxstyle="round,pad=0.5"),
            zorder=100)
    return fig, ax


__all__ = [
    "display_architecture",
    "animate_transport",
    "visualize_reconfiguration",
    "visualize_routing_result",
]
