"""
Stim circuit visualization panels: SVG timeslice, scrolling sidebar, progress bar.

Ported from ``trapped_ion/viz/visualization.py`` to work with the old/
QCCD simulation module.  All rendering uses axes-fraction coordinates
so panels scale correctly inside GridSpec layouts.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

try:
    import matplotlib.pyplot as plt
    import matplotlib.patheffects as path_effects
    from matplotlib.patches import Circle, FancyBboxPatch, Rectangle
    _HAS_MPL = True
except ImportError:
    _HAS_MPL = False

from .constants import (
    BG_COLOR,
    SIDEBAR_BG,
    GATE_KIND_COLORS,
    LASER_MS,
    LASER_ROTATION,
    LASER_MEASURE,
    LASER_RESET,
    PROGRESS_BAR_BG,
    PROGRESS_BAR_COLOR,
    PROGRESS_BAR_EDGE,
)


# =============================================================================
# Data structures
# =============================================================================

@dataclass
class SidebarEntry:
    """A single line in the stim sidebar."""
    line_idx: int = 0
    text: str = ""
    kind: str = ""          # "tick", "gate", "annotation", "comment"
    color: str = "#888888"
    # Legacy fields kept for backward compat
    time: float = 0.0
    gate: str = ""
    qubits: tuple = ()


# =============================================================================
# SVG parser — parse stim timeslice-svg output
# =============================================================================

def parse_stim_timeslice_svg(
    circuit: Any,
    tick: int = 0,
) -> Optional[Dict[str, Any]]:
    """Parse a stim circuit into a timeslice SVG data dict.

    Calls ``circuit.diagram('timeslice-svg', tick=range(tick, tick+1))``
    and regex-parses the resulting SVG string.

    Returns a dict with lists of drawing primitives::

        circles  -> [(cx, cy, r, fill, stroke), ...]
        rects    -> [(x, y, w, h, fill, stroke), ...]
        texts    -> [(x, y, text, font_size, fill), ...]
        paths    -> [(d, stroke, stroke_width, fill), ...]
        viewBox  -> (min_x, min_y, width, height)

    Returns *None* if the circuit has no diagram method or the SVG is empty.
    """
    if circuit is None:
        return None

    diagram_fn = getattr(circuit, "diagram", None)
    if not callable(diagram_fn):
        return None

    try:
        svg_string = str(diagram_fn("timeslice-svg", tick=range(tick, tick + 1)))
    except Exception:
        return None

    if not svg_string or "<svg" not in svg_string.lower():
        return None

    data: Dict[str, Any] = {
        "circles": [],
        "rects": [],
        "texts": [],
        "paths": [],
        "viewBox": (0, 0, 100, 100),
        "qubit_ids": {},  # {(cx, cy): qubit_index}
    }

    # viewBox
    vb = re.search(r'viewBox\s*=\s*"([^"]+)"', svg_string)
    if vb:
        parts = vb.group(1).split()
        if len(parts) == 4:
            data["viewBox"] = tuple(float(p) for p in parts)

    # Circles — also extract qubit index from id="qubit_dot:N:..."
    for m in re.finditer(
            r'<circle[^>]*?'
            r'cx\s*=\s*"([^"]+)"[^>]*?'
            r'cy\s*=\s*"([^"]+)"[^>]*?'
            r'r\s*=\s*"([^"]+)"', svg_string):
        cx, cy, r = float(m.group(1)), float(m.group(2)), float(m.group(3))
        fill = "black"
        stroke = "none"
        fm = re.search(r'fill\s*=\s*"([^"]+)"', m.group(0))
        sm = re.search(r'stroke\s*=\s*"([^"]+)"', m.group(0))
        if fm:
            fill = fm.group(1)
        if sm:
            stroke = sm.group(1)
        # Also check style attribute
        style_m = re.search(r'style\s*=\s*"([^"]*)"', m.group(0))
        if style_m:
            sf = re.search(r'fill\s*:\s*([^;"]+)', style_m.group(1))
            ss = re.search(r'stroke\s*:\s*([^;"]+)', style_m.group(1))
            if sf:
                fill = sf.group(1).strip()
            if ss:
                stroke = ss.group(1).strip()
        data["circles"].append((cx, cy, r, fill, stroke))
        # Extract qubit index from id="qubit_dot:N:..." on small dots
        id_m = re.search(r'id\s*=\s*"qubit_dot:(\d+):', m.group(0))
        if id_m and r < 5:
            data["qubit_ids"][(cx, cy)] = int(id_m.group(1))

    # Rects
    for m in re.finditer(
            r'<rect[^>]*?'
            r'x\s*=\s*"([^"]+)"[^>]*?'
            r'y\s*=\s*"([^"]+)"[^>]*?'
            r'width\s*=\s*"([^"]+)"[^>]*?'
            r'height\s*=\s*"([^"]+)"', svg_string):
        x, y = float(m.group(1)), float(m.group(2))
        w, h = float(m.group(3)), float(m.group(4))
        fill = "none"
        stroke = "none"
        fm = re.search(r'fill\s*=\s*"([^"]+)"', m.group(0))
        sm = re.search(r'stroke\s*=\s*"([^"]+)"', m.group(0))
        if fm:
            fill = fm.group(1)
        if sm:
            stroke = sm.group(1)
        # Also check style attribute
        style_m = re.search(r'style\s*=\s*"([^"]*)"', m.group(0))
        if style_m:
            sf = re.search(r'fill\s*:\s*([^;"]+)', style_m.group(1))
            ss = re.search(r'stroke\s*:\s*([^;"]+)', style_m.group(1))
            if sf:
                fill = sf.group(1).strip()
            if ss:
                stroke = ss.group(1).strip()
        data["rects"].append((x, y, w, h, fill, stroke))

    # Texts — flexible attribute order; handle <tspan> sub-elements
    for m in re.finditer(r'<text([^>]*)>(.*?)</text>', svg_string, re.DOTALL):
        attrs, inner = m.group(1), m.group(2)
        xm = re.search(r'\bx\s*=\s*"([^"]+)"', attrs)
        ym = re.search(r'\by\s*=\s*"([^"]+)"', attrs)
        fsm = re.search(r'font-size\s*=\s*"([^"]+)"', attrs)
        fm = re.search(r'\bfill\s*=\s*"([^"]+)"', attrs)
        if not (xm and ym and fsm):
            continue
        tx, ty = float(xm.group(1)), float(ym.group(1))
        fs = float(fsm.group(1))
        fill = fm.group(1) if fm else "black"
        # Strip HTML tags (<tspan ...>X</tspan> → X) and join text
        txt = re.sub(r'<[^>]+>', '', inner).strip()
        if txt:
            data["texts"].append((tx, ty, txt, fs, fill))

    # Paths
    for m in re.finditer(
            r'<path[^>]*?d\s*=\s*"([^"]+)"', svg_string):
        d = m.group(1)
        stroke = "black"
        sw = 1.0
        fill = "none"
        sm = re.search(r'stroke\s*=\s*"([^"]+)"', m.group(0))
        wm = re.search(r'stroke-width\s*=\s*"([^"]+)"', m.group(0))
        fm = re.search(r'fill\s*=\s*"([^"]+)"', m.group(0))
        if sm:
            stroke = sm.group(1)
        if wm:
            sw = float(wm.group(1))
        if fm:
            fill = fm.group(1)
        data["paths"].append((d, stroke, sw, fill))

    # ── Content-tight viewBox ────────────────────────────────────
    # Recompute the viewBox from actual drawn primitives (circles,
    # rects, texts) rather than trusting stim's full-grid viewBox.
    # This produces a much tighter bounding box when the circuit has
    # multi-block vertical stacking with large gaps.
    all_xs: List[float] = []
    all_ys: List[float] = []
    for cx, cy, r, _f, _s in data["circles"]:
        all_xs.extend([cx - r, cx + r])
        all_ys.extend([cy - r, cy + r])
    for rx, ry, rw, rh, _f, _s in data["rects"]:
        all_xs.extend([rx, rx + rw])
        all_ys.extend([ry, ry + rh])
    for tx, ty, _txt, fs, _f in data["texts"]:
        all_xs.append(tx)
        all_ys.append(ty)
    if all_xs and all_ys:
        pad = 8.0  # SVG-coordinate units of padding
        tight_vb = (
            min(all_xs) - pad,
            min(all_ys) - pad,
            max(all_xs) - min(all_xs) + 2 * pad,
            max(all_ys) - min(all_ys) + 2 * pad,
        )
        # Use the tighter box only if it's meaningfully smaller
        orig_area = data["viewBox"][2] * data["viewBox"][3]
        tight_area = tight_vb[2] * tight_vb[3]
        if tight_area < orig_area * 0.95:
            data["viewBox"] = tight_vb

    return data


# =============================================================================
# SVG renderer — draw parsed SVG primitives onto matplotlib axes
# =============================================================================

def draw_stim_svg(
    ax: Any,
    svg_data: Dict[str, Any],
    title: str = "Timeslice",
) -> None:
    """Render parsed SVG primitives onto a matplotlib axes.

    Handles two stim SVG styles:
    * **Gate boxes** (R, H, M, …): ``<rect>`` + ``<text>`` — coloured
      FancyBboxPatch with the gate name rendered in white.
    * **CX / CNOT**: ``<circle r≥8>`` (control = filled black,
      target = open white) connected by ``<path>`` lines.  Small
      ``<circle r≤3>`` qubit-position dots are rendered as discrete
      grey markers.

    Uses axes-fraction coordinates so the drawing scales to whatever
    panel size GridSpec assigns.
    """
    if not _HAS_MPL or svg_data is None:
        return

    ax.clear()
    ax.set_facecolor(BG_COLOR)
    ax.axis("off")

    vb = svg_data["viewBox"]
    vb_x, vb_y, vb_w, vb_h = vb
    margin = 0.08
    span = 1.0 - 2 * margin

    # Scale to fit the larger dimension while preserving aspect ratio
    scale_x = span / max(vb_w, 1)
    scale_y = span / max(vb_h, 1)
    scale = min(scale_x, scale_y)
    off_x = margin + (span - vb_w * scale) / 2
    off_y = margin + (span - vb_h * scale) / 2

    def _tx(sx: float) -> float:
        return off_x + (sx - vb_x) * scale

    def _ty(sy: float) -> float:
        return 1.0 - off_y - (sy - vb_y) * scale  # Y flipped

    ax.set_title(title, fontsize=18, fontweight="bold", pad=8)

    # Gate-name → colour mapping for SVG boxes
    _SVG_GATE_COLORS: Dict[str, str] = {
        "H": "#4CAF50", "S": "#7C4DFF", "S_DAG": "#7C4DFF",
        "T": "#7C4DFF", "X": "#F44336", "Y": "#FF9800",
        "Z": "#2196F3", "I": "#9E9E9E",
        "CX": "#1565C0", "CZ": "#00897B", "CNOT": "#1565C0",
        "M": "#FF5722", "MR": "#FF5722", "MZ": "#FF5722",
        "MX": "#FF5722", "MY": "#FF5722", "MRX": "#FF5722",
        "MRZ": "#FF5722", "MRY": "#FF5722",
        "R": "#00BCD4", "RX": "#00BCD4", "RZ": "#00BCD4",
        "RY": "#00BCD4", "SQRT_X": "#AB47BC",
        "SQRT_XX": "#FFB300", "SPP": "#FFB300",
    }
    CX_COLOR = "#1565C0"   # blue for CNOT

    # Pre-index text labels by approximate position so gate boxes can
    # be coloured by the nearest label name.
    _text_by_pos: List[Tuple[float, float, str]] = []
    for tx_v, ty_v, txt, _fs, _fill in svg_data["texts"]:
        _text_by_pos.append((_tx(tx_v), _ty(ty_v), txt.strip().upper()))

    def _nearest_gate_color(cx_pos: float, cy_pos: float) -> Tuple[str, str]:
        """Find the gate label closest to (cx_pos, cy_pos).

        Returns ``(gate_name, colour)``.
        """
        best_d = float("inf")
        best_name = ""
        for tx, ty, name in _text_by_pos:
            d = abs(tx - cx_pos) + abs(ty - cy_pos)
            if d < best_d:
                best_d = d
                best_name = name
        if best_name and best_d < 0.40:
            # Try progressively shorter prefixes for compound names
            # e.g. "MRX" → "MR" → "M"
            color = _SVG_GATE_COLORS.get(best_name)
            if color is None and len(best_name) > 2:
                color = _SVG_GATE_COLORS.get(best_name[:2])
            if color is None and len(best_name) > 1:
                color = _SVG_GATE_COLORS.get(best_name[:1])
            return (best_name, color or "#5C6BC0")
        return ("", "#5C6BC0")  # indigo default

    # --- Classify circles: small qubit dots vs large gate symbols ---
    small_circles = []   # r < 5  (qubit position markers)
    large_circles = []   # r >= 5 (CX control/target symbols)
    for cx, cy, r, fill, stroke in svg_data["circles"]:
        if r >= 5:
            large_circles.append((cx, cy, r, fill, stroke))
        else:
            small_circles.append((cx, cy, r, fill, stroke))

    # Detect CX mode: if we have large circles but no rects/texts,
    # this TICK contains CX/CNOT gates.
    is_cx_tick = bool(large_circles) and not svg_data["rects"]

    # --- Compute qubit wire positions from small circles ---
    _all_qubit_ys: List[float] = []
    for cx_c, cy_c, r, fill, stroke in small_circles:
        _all_qubit_ys.append(_ty(cy_c))
    if not _all_qubit_ys and svg_data["rects"]:
        for x, y, w, h, fill, stroke in svg_data["rects"]:
            _all_qubit_ys.append(_ty(y + h / 2))
    if not _all_qubit_ys and large_circles:
        for cx_c, cy_c, r, fill, stroke in large_circles:
            _all_qubit_ys.append(_ty(cy_c))
    qubit_ys = sorted(set(round(y, 4) for y in _all_qubit_ys))

    # Draw horizontal qubit wire lines (circuit-diagram style)
    if qubit_ys:
        wire_x0 = margin * 0.5
        wire_x1 = 1.0 - margin * 0.5
        for qy in qubit_ys:
            ax.plot([wire_x0, wire_x1], [qy, qy],
                    color="#888", linewidth=0.7, alpha=0.35,
                    transform=ax.transAxes, zorder=1, clip_on=False)

    # --- Compute adaptive gate-box size from qubit spacing ---
    n_rects = len(svg_data["rects"])
    n_wires = len(qubit_ys)
    qubit_spacing = 0.12  # default for sparse / unknown layouts
    if n_wires >= 2:
        _qy_diffs = [abs(qubit_ys[i + 1] - qubit_ys[i])
                      for i in range(n_wires - 1)]
        qubit_spacing = min(_qy_diffs) if _qy_diffs else 0.12
        # Box height must fit within wire spacing to avoid overlap
        gate_box_h = max(0.012, min(qubit_spacing * 0.70, 0.11))
        gate_box_w = gate_box_h * 1.3  # slightly wider than tall
    elif n_rects > 0:
        gate_box_w = max(0.05, min(0.12, 0.55 / max(n_rects, 2)))
        gate_box_h = max(0.04, min(0.10, 0.45 / max(n_rects, 2)))
    else:
        gate_box_w = 0.10
        gate_box_h = 0.08

    # Pre-associate each rect with its nearest text so we can colour+label it
    rect_text_map: Dict[int, Tuple[str, str]] = {}  # rect_idx -> (gate_name, color)
    for ri, (x, y, w, h, fill, stroke) in enumerate(svg_data["rects"]):
        cx_r = _tx(x + w / 2)
        cy_r = _ty(y + h / 2)
        gate_name, color = _nearest_gate_color(cx_r, cy_r)
        rect_text_map[ri] = (gate_name, color)

    # --- Rectangles (gate boxes — quantum circuit style) ---
    for ri, (x, y, w, h, fill, stroke) in enumerate(svg_data["rects"]):
        cx_r = _tx(x + w / 2)
        cy_r = _ty(y + h / 2)
        gate_name, box_fill = rect_text_map.get(ri, ("", "#5C6BC0"))
        # Widen box for long gate names (MRX, SQRT_X, etc.)
        name_len = len(gate_name) if gate_name else 1
        box_w = gate_box_w * max(1.0, name_len * 0.55) if name_len > 2 else gate_box_w
        rect = FancyBboxPatch(
            (cx_r - box_w / 2, cy_r - gate_box_h / 2),
            box_w, gate_box_h,
            boxstyle="round,pad=0.006",
            facecolor=box_fill,
            edgecolor="white",
            linewidth=1.5, alpha=0.95,
            transform=ax.transAxes, zorder=3, clip_on=False,
        )
        ax.add_patch(rect)
        # Draw gate name label centered on box
        if gate_name:
            lbl_fs = max(9, min(22, gate_box_h * 200))
            # Shrink font for long names
            if name_len > 2:
                lbl_fs = max(5, lbl_fs * 0.85)
            ax.text(cx_r, cy_r, gate_name,
                    fontsize=lbl_fs, fontfamily="monospace",
                    fontweight="bold", color="white",
                    ha="center", va="center",
                    transform=ax.transAxes, zorder=5, clip_on=False,
                    path_effects=[path_effects.withStroke(
                        linewidth=3, foreground="#222")])

    # --- Large circles (CX control/target) ---
    # Scale CX dot radius with qubit spacing to prevent overlap
    cx_dot_r = min(0.035, max(0.010, qubit_spacing * 0.40))
    for cx, cy, r, fill, stroke in large_circles:
        mcx = _tx(cx)
        mcy = _ty(cy)
        is_target = (fill in ("white", "#fff", "#ffffff", "none")
                     and stroke not in ("none", ""))
        if is_target:
            # CNOT target: open circle with ⊕
            circ = Circle(
                (mcx, mcy), cx_dot_r,
                facecolor="white",
                edgecolor=CX_COLOR,
                linewidth=2.5,
                transform=ax.transAxes, zorder=5, clip_on=False,
            )
            ax.add_patch(circ)
            # Draw ⊕ cross inside
            arm = cx_dot_r * 0.75
            ax.plot([mcx - arm, mcx + arm], [mcy, mcy],
                    color=CX_COLOR, linewidth=2.0,
                    transform=ax.transAxes, zorder=6, clip_on=False)
            ax.plot([mcx, mcx], [mcy - arm, mcy + arm],
                    color=CX_COLOR, linewidth=2.0,
                    transform=ax.transAxes, zorder=6, clip_on=False)
        else:
            # CNOT control: filled circle
            circ = Circle(
                (mcx, mcy), cx_dot_r * 0.55,
                facecolor=CX_COLOR,
                edgecolor=CX_COLOR,
                linewidth=1.5,
                transform=ax.transAxes, zorder=5, clip_on=False,
            )
            ax.add_patch(circ)

    # --- Small circles (qubit position dots) ---
    small_dot_r = min(0.012, max(0.003, qubit_spacing * 0.12))
    for cx, cy, r, fill, stroke in small_circles:
        mcx = _tx(cx)
        mcy = _ty(cy)
        circ = Circle(
            (mcx, mcy), small_dot_r,
            facecolor="#BDBDBD",
            edgecolor="#999",
            linewidth=0.8,
            transform=ax.transAxes, zorder=2, clip_on=False,
        )
        ax.add_patch(circ)

    # --- Qubit index labels (from SVG circle IDs) ---
    # Label every qubit at its actual (x, y) position, not once per wire.
    # Skip labels when circuit is too dense to avoid unreadable clutter.
    _qubit_id_map = svg_data.get("qubit_ids", {})
    _show_qubit_labels = len(_qubit_id_map) <= 40
    if _show_qubit_labels:
        label_fs = max(8, min(16, gate_box_h * 180))
        lbl_stroke = max(1.5, min(2.5, gate_box_h * 50))
        for (cx_q, cy_q), qidx in _qubit_id_map.items():
            lx = _tx(cx_q)
            ly = _ty(cy_q)
            # Place label above the qubit dot
            ax.text(lx, ly + gate_box_h * 0.8, f"q{qidx}",
                    fontsize=label_fs, fontfamily="monospace",
                    fontweight="bold", color="#444",
                    ha="center", va="bottom",
                    transform=ax.transAxes, zorder=7, clip_on=False,
                    path_effects=[path_effects.withStroke(
                        linewidth=lbl_stroke, foreground="white")])

    # --- Text labels (gate names) ---
    # Gate name labels are now rendered directly on the gate boxes above.
    # Skip the separate text pass to avoid double-rendering.
    # (The text data is still used by _nearest_gate_color for colour lookup.)

    # --- Paths (CX connecting lines / qubit wires) ---
    for d_str, stroke, sw, fill in svg_data["paths"]:
        if stroke == "none" and fill == "none":
            continue
        segs = re.findall(
            r'([MLHVCSQTAZ])\s*([\d\s.,e+-]*)', d_str, re.IGNORECASE)
        px, py = 0.0, 0.0
        pts_x: List[float] = []
        pts_y: List[float] = []
        for cmd, args in segs:
            nums = [float(n) for n in re.findall(r'[\d.eE+-]+', args)]
            cu = cmd.upper()
            if cu == "M":
                if len(nums) >= 2:
                    if cmd == "M":
                        px, py = nums[0], nums[1]
                    else:
                        px += nums[0]
                        py += nums[1]
                    if pts_x:
                        lw_draw = max(1.5, sw * scale * 12) if is_cx_tick else max(0.8, sw * scale * 8)
                        ax.plot(pts_x, pts_y,
                                color=CX_COLOR if is_cx_tick else (stroke if stroke != "none" else "#555"),
                                linewidth=lw_draw,
                                transform=ax.transAxes,
                                solid_capstyle="round", zorder=3,
                                clip_on=False)
                    pts_x, pts_y = [_tx(px)], [_ty(py)]
            elif cu == "L":
                for j in range(0, len(nums) - 1, 2):
                    if cmd == "L":
                        px, py = nums[j], nums[j + 1]
                    else:
                        px += nums[j]
                        py += nums[j + 1]
                    pts_x.append(_tx(px))
                    pts_y.append(_ty(py))
            elif cu == "H":
                for j in range(len(nums)):
                    px = nums[j] if cmd == "H" else px + nums[j]
                    pts_x.append(_tx(px))
                    pts_y.append(_ty(py))
            elif cu == "V":
                for j in range(len(nums)):
                    py = nums[j] if cmd == "V" else py + nums[j]
                    pts_x.append(_tx(px))
                    pts_y.append(_ty(py))

        # Draw remaining path segment
        if len(pts_x) >= 2:
            lw_draw = max(1.5, sw * scale * 12) if is_cx_tick else max(0.8, sw * scale * 8)
            ax.plot(pts_x, pts_y,
                    color=CX_COLOR if is_cx_tick else (stroke if stroke != "none" else "#555"),
                    linewidth=lw_draw,
                    transform=ax.transAxes,
                    solid_capstyle="round", zorder=3, clip_on=False)

    # --- CX label annotation ---
    if is_cx_tick and large_circles:
        # Place a "CX" label in the top-left area
        ax.text(0.05, 0.92, "CNOT",
                fontsize=16, fontfamily="monospace", fontweight="bold",
                color=CX_COLOR, alpha=0.85,
                ha="left", va="top",
                transform=ax.transAxes, zorder=10, clip_on=False)


# =============================================================================
# Sidebar parser — split stim circuit into annotated lines
# =============================================================================

def parse_stim_for_sidebar(
    circuit: Any,
) -> List[SidebarEntry]:
    """Parse a stim circuit into sidebar entries (one per line).

    Classifies each line as 'tick', 'gate', 'annotation', or 'comment'.
    """
    if circuit is None:
        return []

    try:
        text = str(circuit)
    except Exception:
        return []

    lines = text.splitlines()
    entries: List[SidebarEntry] = []

    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            continue

        kind = "gate"
        color = "#888888"

        upper = stripped.upper()
        if upper.startswith("TICK"):
            kind = "tick"
            color = "#1565C0"
        elif upper.startswith("#"):
            kind = "comment"
            color = "#9E9E9E"
        elif upper.startswith(("DETECTOR", "OBSERVABLE_INCLUDE")):
            kind = "annotation"
            color = "#6A1B9A"
        elif upper.startswith("REPEAT") or stripped == "}":
            kind = "annotation"
            color = "#795548"
        elif upper.startswith(("M", "MR", "MX", "MY", "MZ")):
            kind = "gate"
            color = LASER_MEASURE
        elif upper.startswith(("R", "RX", "RY", "RZ")):
            kind = "gate"
            color = LASER_ROTATION
        elif any(upper.startswith(g) for g in
                 ("CX", "CZ", "CNOT", "XCX", "XCZ", "ZCX", "ZCZ",
                  "ISWAP", "SQRT_XX", "SQRT_ZZ", "SPP")):
            kind = "gate"
            color = LASER_MS
        elif upper.startswith("QUBIT_COORDS"):
            kind = "annotation"
            color = "#795548"
        elif upper.startswith("SHIFT_COORDS"):
            kind = "annotation"
            color = "#795548"
        elif any(upper.startswith(g) for g in
                 ("H", "S", "X", "Y", "Z", "I", "SQRT",
                  "S_DAG", "DEPOLARIZE", "X_ERROR", "Z_ERROR")):
            kind = "gate"
            color = LASER_ROTATION

        # Parse gate name and qubit targets from the instruction text
        gate_name = stripped.split()[0] if stripped else ""
        qubits: tuple = ()
        if kind == "gate" and gate_name:
            # Stim instruction format: GATE_NAME target1 target2 ...
            # Targets are integers; skip any parenthesised arguments.
            parts = stripped.split()
            _q_list: list = []
            for p in parts[1:]:
                # Skip parenthesized noise args like "(0.001)"
                if p.startswith("("):
                    continue
                try:
                    _q_list.append(int(p))
                except ValueError:
                    pass
            qubits = tuple(_q_list)

        entries.append(SidebarEntry(
            line_idx=i,
            text=stripped,
            kind=kind,
            color=color,
            gate=gate_name,
            qubits=qubits,
        ))

    return entries


# =============================================================================
# Sidebar renderer — scrolling stim circuit text
# =============================================================================

def draw_sidebar(
    ax: Any,
    entries: Sequence[SidebarEntry],
    highlight_lines: Optional[Set[int]] = None,
    center_idx: int = 0,
    title: str = "Stim Circuit",
) -> None:
    """Draw scrolling stim sidebar with highlighted current lines.

    Only gate instructions are displayed (QUBIT_COORDS, TICK, DETECTOR,
    OBSERVABLE_INCLUDE are filtered out).  The highlight indices still
    reference positions in the *full* entries list so the mapping logic
    works unchanged.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
    entries : list of SidebarEntry
    highlight_lines : set of line indices to highlight (yellow bg)
    center_idx : line index for the pointer arrow (orange)
    title : sidebar title
    """
    if not _HAS_MPL:
        return
    if highlight_lines is None:
        highlight_lines = set()

    ax.clear()
    ax.set_facecolor(SIDEBAR_BG)
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title(title, fontsize=18, fontweight="bold", pad=8)

    if not entries:
        return

    # Filter to gate instructions only (skip QUBIT_COORDS, TICK, DETECTOR, etc.)
    gate_entries = [
        (orig_idx, e) for orig_idx, e in enumerate(entries)
        if e.kind == "gate"
    ]
    if not gate_entries:
        return

    n_lines = len(gate_entries)
    window = min(6, n_lines)  # tight sliding window – only 6 lines

    # Find the filtered position of the highlighted / center entry
    filtered_center = 0
    for fi, (orig_idx, _) in enumerate(gate_entries):
        if orig_idx in highlight_lines:
            filtered_center = fi
            break
        if orig_idx >= center_idx:
            filtered_center = fi
            break

    w_start = max(0, filtered_center - window // 2)
    w_end = min(n_lines, w_start + window)
    if w_end - w_start < window:
        w_start = max(0, w_end - window)

    # Compact vertical layout: leave room for title at top and
    # position/total counter at bottom.
    y_top = 0.88          # just below the title
    y_bot = 0.10          # leave room for counter text
    usable = y_top - y_bot
    row_h = usable / window if window else usable

    for vi, fi in enumerate(range(w_start, w_end)):
        orig_idx, entry = gate_entries[fi]
        is_highlighted = orig_idx in highlight_lines
        y_frac = y_top - (vi + 0.5) * row_h

        # Yellow highlight background
        if is_highlighted:
            ax.axhspan(
                y_frac - 0.5 * row_h,
                y_frac + 0.5 * row_h,
                facecolor="#FFD600", alpha=0.45, zorder=0)

        # Orange pointer arrow on highlighted line
        if is_highlighted and highlight_lines:
            ax.text(0.97, y_frac, "\u25C0",
                    transform=ax.transAxes,
                    fontsize=18, color="#E65100",
                    fontweight="bold", ha="right", va="center",
                    zorder=5)

        # Coloured dot for gate kind
        ax.plot(0.03, y_frac, "o",
                color=entry.color, markersize=7,
                transform=ax.transAxes, zorder=4, clip_on=False)

        # Truncated line text – larger font for 6-line window
        trunc = entry.text[:46]
        if len(entry.text) > 46:
            trunc += "\u2026"
        ax.text(0.08, y_frac, trunc,
                transform=ax.transAxes,
                fontsize=16, fontfamily="monospace",
                color="#000" if is_highlighted else "#555",
                fontweight="bold" if is_highlighted else "normal",
                va="center", zorder=3)

    # Position counter at the bottom of the sidebar
    ax.text(0.50, 0.03,
            f"{filtered_center + 1} / {n_lines}",
            transform=ax.transAxes,
            fontsize=14, fontfamily="monospace",
            ha="center", va="bottom",
            color="#888", zorder=3)


def draw_ops_sidebar(
    ax: Any,
    labels: List[str],
    gate_kinds: List[Optional[str]],
    is_transport: List[bool],
    current_step: int = 0,
    title: str = "Timeslice  (steps)",
) -> None:
    """Draw ops-list sidebar (fallback when no stim circuit).

    Parameters
    ----------
    ax : matplotlib.axes.Axes
    labels : step label strings
    gate_kinds : gate kind per step (or None for transport)
    is_transport : True if step is transport
    current_step : currently highlighted step index
    title : sidebar title
    """
    if not _HAS_MPL:
        return

    ax.clear()
    ax.set_facecolor(SIDEBAR_BG)
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title(title, fontsize=16, fontweight="bold", pad=6)

    n_labels = len(labels)
    if n_labels == 0:
        return

    kind_colors = {
        "ms": LASER_MS, "rotation": LASER_ROTATION,
        "measure": LASER_MEASURE, "reset": LASER_RESET,
        None: "#78909C",
    }

    window = min(22, n_labels)
    w_start = max(0, current_step - window // 2)
    w_end = min(n_labels, w_start + window)
    if w_end - w_start < window:
        w_start = max(0, w_end - window)

    for vi, sidx in enumerate(range(w_start, w_end)):
        is_cur = sidx == current_step
        y_frac = 1.0 - (vi + 0.5) / window
        gk = gate_kinds[sidx] if sidx < len(gate_kinds) else None
        is_xport = is_transport[sidx] if sidx < len(is_transport) else False

        # Highlight current step
        if is_cur:
            ax.axhspan(y_frac - 0.5 / window,
                       y_frac + 0.5 / window,
                       facecolor="#FFD600", alpha=0.40, zorder=0)
            ax.text(0.98, y_frac, "\u25C0",
                    transform=ax.transAxes,
                    fontsize=14, color="#E65100",
                    fontweight="bold", ha="right", va="center",
                    zorder=5)

        # Coloured dot
        dot_c = kind_colors.get(gk, "#78909C")
        if is_xport:
            dot_c = "#42A5F5"
        ax.plot(0.04, y_frac, "o",
                color=dot_c, markersize=5.5,
                transform=ax.transAxes, zorder=4, clip_on=False)

        # Step label
        lbl = labels[sidx] if sidx < len(labels) else f"Step {sidx}"
        trunc = lbl[:44]
        ax.text(0.08, y_frac,
                f"{sidx:3d}\u2502 {trunc}",
                transform=ax.transAxes,
                fontsize=12, fontfamily="monospace",
                color="#000" if is_cur else "#666",
                fontweight="bold" if is_cur else "normal",
                va="center", zorder=3)


# =============================================================================
# Progress bar
# =============================================================================

def draw_progress_bar(
    ax: Any,
    progress_frac: float,
    elapsed_us: float = 0.0,
    total_us: float = 0.0,
    step: int = 0,
    total_steps: int = 0,
) -> None:
    """Draw a progress bar at the bottom of the architecture panel.

    Uses axes-fraction coordinates (overlaid on the arch axes).

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The architecture axes (bar is drawn in axes-fraction space).
    progress_frac : float
        Progress 0.0 to 1.0.
    elapsed_us : float
        Elapsed physical time in microseconds.
    total_us : float
        Total physical time in microseconds.
    step : int
        Current step number.
    total_steps : int
        Total number of steps.
    """
    if not _HAS_MPL:
        return

    bar_width = 0.60
    bar_height = 0.022
    bar_x = 0.20
    bar_y = 0.96          # TOP of axes — between title and QCCD arch

    # Background bar
    bg_rect = Rectangle(
        (bar_x, bar_y), bar_width, bar_height,
        transform=ax.transAxes,
        facecolor=PROGRESS_BAR_BG,
        edgecolor=PROGRESS_BAR_EDGE,
        linewidth=1.0, zorder=99, clip_on=False,
    )
    ax.add_patch(bg_rect)

    # Progress fill
    fill_rect = Rectangle(
        (bar_x, bar_y), bar_width * max(0, min(1, progress_frac)), bar_height,
        transform=ax.transAxes,
        facecolor=PROGRESS_BAR_COLOR,
        edgecolor="none", zorder=100, clip_on=False,
    )
    ax.add_patch(fill_rect)

    # Time label (below the bar so it doesn't overlap the title)
    pct_str = f"{progress_frac * 100:.0f}%"
    step_str = f"Step {step}/{total_steps}" if total_steps > 0 else ""
    if total_us > 0:
        time_str = f"{elapsed_us:.0f}/{total_us:.0f} \u00b5s  ({pct_str})  {step_str}"
    else:
        # No timing data — show step-based progress only
        time_str = f"{step_str}  ({pct_str})" if step_str else pct_str
    ax.text(bar_x + bar_width / 2, bar_y - 0.008,
            time_str,
            transform=ax.transAxes,
            fontsize=13, ha="center", va="top",
            fontfamily="monospace", fontweight="bold", color="#424242",
            zorder=100, clip_on=False)


# =============================================================================
# Convenience: render a stim timeslice diagram standalone
# =============================================================================

def render_stim_timeslice(
    circuit: Any,
    tick: int = 0,
    title: str = "",
    figsize: Tuple[float, float] = (6, 5),
) -> Any:
    """Render a stim circuit timeslice as a standalone figure.

    Returns the matplotlib Figure, or None on failure.
    """
    if not _HAS_MPL:
        return None

    svg_data = parse_stim_timeslice_svg(circuit, tick=tick)
    if svg_data is None:
        return None

    fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=120)
    draw_stim_svg(ax, svg_data, title=title or f"Timeslice TICK {tick}")
    fig.tight_layout()
    return fig
