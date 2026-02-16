"""
Architecture-agnostic layout data structures and extractors.

Provides the ``_GraphLayout`` dataclass that decouples geometry extraction
from rendering so the same renderer can handle QCCDArch, QCCDWiseArch, or
any future architecture using old/ node types.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .constants import (
    ION_RADIUS,
    JUNCTION_SIDE,
    SPACING,
    ION_SPACING_RATIO,
    TRAP_PAD_X,
    TRAP_PAD_Y,
)


# =============================================================================
# Coordinate normalization
# =============================================================================

def _is_wise_arch(arch: Any) -> bool:
    """Return True if *arch* is a WISE-type architecture."""
    names = tuple(cls.__name__ for cls in type(arch).__mro__)
    return "QCCDWiseArch" in names or "WISEArchitecture" in names


def _detect_arch_scale(arch: Any) -> float:
    """Detect the coordinate scale used by the old architecture.

    Returns a divisor to normalize coordinates to our target SPACING.
    Old architectures use ~60-120 unit grid spacing while we want ~3.8.

    For WISE architectures, callers should use ``_wise_normalizer`` instead;
    this function is only used for augmented-grid / generic arches.
    """
    # Collect all unique positions
    positions = []
    for trap in getattr(arch, "_manipulationTraps", []):
        if hasattr(trap, "pos"):
            positions.append(trap.pos)
    for junc in getattr(arch, "_junctions", []):
        positions.append(junc.pos)
    
    if len(positions) < 2:
        return 1.0  # Can't detect, no scaling
    
    # Find minimum non-zero distance between positions
    min_dist = float('inf')
    for i, p1 in enumerate(positions):
        for p2 in positions[i+1:]:
            dx = abs(float(p1[0]) - float(p2[0]))
            dy = abs(float(p1[1]) - float(p2[1]))
            d = max(dx, dy)  # Use Manhattan-ish distance
            if d > 5:  # Ignore tiny differences
                min_dist = min(min_dist, d)
    
    if min_dist == float('inf') or min_dist < 5:
        return 1.0
    
    # Target: node spacing of ~8 units so traps have room to be visible
    # With SPACING=3.8 and traps ~2 units wide, we want ~8 unit gaps
    target_spacing = 8.0
    return min_dist / target_spacing


def _wise_normalizer(arch: Any):
    """Return a position-normalizer ``(sx, sy)`` for WISE architectures.

    Maps raw arch coordinates to the synthetic WISE layout space used by
    ``_extract_wise_layout`` / ``_draw_wise_grid``.

    The arch places traps at grid ``(2*col, row)`` mapped through
    ``_gridToCoordinate → pos * (k+1)``.  The WISE layout uses
    ``(col * block_pitch, row * row_pitch)``.  The linear transform is::

        wise_x = arch_x * block_pitch / (2 * (k+1))
        wise_y = arch_y * row_pitch / (k+1)
    """
    _cfg = getattr(arch, "wise_config", None) or arch
    k = getattr(_cfg, "ions_per_segment", getattr(_cfg, "k", 2))
    ion_sp = SPACING * ION_SPACING_RATIO
    trap_inner_w = (k - 1) * ion_sp
    trap_w = trap_inner_w + 2 * TRAP_PAD_X
    junc_gap = SPACING * 1.2
    block_pitch = trap_w + junc_gap
    row_pitch = SPACING * 2.2
    kp1 = k + 1
    sx = block_pitch / (2.0 * kp1)
    sy = row_pitch / kp1
    return sx, sy


def _normalize_pos(pos: Tuple[Any, Any], scale: float) -> Tuple[float, float]:
    """Normalize a position tuple by the detected scale."""
    return (float(pos[0]) / scale, float(pos[1]) / scale)


# =============================================================================
# Layout data structures
# =============================================================================

@dataclass
class _TrapInfo:
    """Position and content of one trap segment."""
    cx: float
    cy: float
    width: float
    height: float
    label: str
    ion_indices: List[int]
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
    ion_positions: Dict[int, Tuple[float, float]]
    info_lines: List[str]
    auto_title: str


# =============================================================================
# WISE layout extractor  (for QCCDWiseArch)
# =============================================================================

def _extract_wise_layout(arch: Any, ion_sp: float) -> _GraphLayout:
    """Build a ``_GraphLayout`` from a WISE architecture (QCCDWiseArch).

    Parameters
    ----------
    arch : QCCDWiseArch
        Must expose ``rows``, ``ions_per_segment`` (or ``k``),
        ``col_groups`` (or ``m``), ``total_columns``.
    ion_sp : float
        Inter-ion spacing inside a trap.
    """
    # WISEArchitecture wraps QCCDWiseArch as arch.wise_config.
    # Chain through to find m, n, k from the inner config if needed.
    _cfg = getattr(arch, "wise_config", None) or arch
    rows = getattr(_cfg, "rows", getattr(_cfg, "n", 1))
    k = getattr(_cfg, "ions_per_segment", getattr(_cfg, "k", 2))
    m = getattr(_cfg, "col_groups", getattr(_cfg, "m", 1))
    total_cols = getattr(_cfg, "total_columns", m * k)

    trap_inner_w = (k - 1) * ion_sp
    trap_w = trap_inner_w + 2 * TRAP_PAD_X
    trap_h = 2 * ION_RADIUS + 2 * TRAP_PAD_Y
    junc_gap = SPACING * 1.2
    block_pitch = trap_w + junc_gap
    row_pitch = SPACING * 2.2

    def _trap_xy(b: int, r: int) -> Tuple[float, float]:
        return (b * block_pitch, r * row_pitch)

    def _junc_xy(b: int, r: int) -> Tuple[float, float]:
        tx, _ = _trap_xy(b, r)
        return (tx + trap_w / 2 + junc_gap / 2, r * row_pitch)

    def _ion_xy(b: int, r: int, slot: int) -> Tuple[float, float]:
        tx, ty = _trap_xy(b, r)
        return (tx - trap_inner_w / 2 + slot * ion_sp, ty)

    traps: List[_TrapInfo] = []
    junctions: List[_JunctionInfo] = []
    edges: List[_EdgeInfo] = []
    ion_positions: Dict[int, Tuple[float, float]] = {}

    # --- Traps & ions ---
    for b in range(m):
        for r in range(rows):
            cx, cy = _trap_xy(b, r)
            ion_ids: List[int] = []
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

    # --- Junctions ---
    for b in range(m - 1):
        for r in range(rows):
            jx, jy = _junc_xy(b, r)
            junctions.append(_JunctionInfo(jx, jy, f"J({b},{r})"))

    # --- Horizontal crossing edges ---
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

    # --- Vertical dashed links between junctions ---
    for r in range(rows - 1):
        for b in range(m - 1):
            jx1, jy1 = _junc_xy(b, r)
            jx2, jy2 = _junc_xy(b, r + 1)
            edges.append(_EdgeInfo(
                jx1, jy1 + JUNCTION_SIDE / 2,
                jx2, jy2 - JUNCTION_SIDE / 2,
                dashed=True,
            ))

    # --- Info ---
    n_ions = getattr(arch, "num_qubits", m * k * rows)
    info_lines = [
        f"m={m}  n={rows}  k={k}",
        f"Ions: {n_ions}   Traps: {m * rows}",
        f"Junctions: {(m - 1) * rows}",
    ]
    auto_title = (
        f"WISE  m={m} \u00d7 n={rows} \u00d7 k={k}  [{n_ions} ions]"
    )
    return _GraphLayout(
        traps=traps, junctions=junctions, edges=edges,
        ion_positions=ion_positions, info_lines=info_lines,
        auto_title=auto_title,
    )


# =============================================================================
# Generic QCCD layout extractor  (for QCCDArch)
# =============================================================================

def _extract_qccd_layout(arch: Any, ion_sp: float) -> _GraphLayout:
    """Build a ``_GraphLayout`` from a generic ``QCCDArch`` instance.

    Works with old/'s node types: ``ManipulationTrap``, ``Junction``,
    ``Crossing``, ``Ion`` accessed via ``arch._manipulationTraps``,
    ``arch._junctions``, ``arch._crossings``.
    """
    traps: List[_TrapInfo] = []
    junctions: List[_JunctionInfo] = []
    edges: List[_EdgeInfo] = []
    ion_positions: Dict[int, Tuple[float, float]] = {}

    # --- Traps ---
    trap_list = getattr(arch, "_manipulationTraps", [])
    for trap in trap_list:
        # Skip non-trap entries (some are edge-list tuples)
        if not hasattr(trap, "pos"):
            continue
        px, py = trap.pos
        ions = list(getattr(trap, "ions", []))
        ni = len(ions)
        cap = getattr(trap, "capacity", ni or 3)
        horiz = getattr(trap, "isHorizontal", True)
        spacing = getattr(trap, "spacing", ion_sp)

        inner_span = max(0, (ni - 1)) * spacing
        if horiz:
            tw = inner_span + 2 * TRAP_PAD_X
            th = 2 * ION_RADIUS + 2 * TRAP_PAD_Y
        else:
            tw = 2 * ION_RADIUS + 2 * TRAP_PAD_Y
            th = inner_span + 2 * TRAP_PAD_X

        ion_ids: List[int] = []
        for i, ion in enumerate(ions):
            ix_pos, iy_pos = ion.pos
            ion_positions[ion.idx] = (ix_pos, iy_pos)
            ion_ids.append(ion.idx)

        label_str = getattr(trap, "label", f"T{trap.idx}")
        traps.append(_TrapInfo(
            cx=px, cy=py, width=tw, height=th,
            label=f"{label_str} k={cap}",
            ion_indices=ion_ids, is_horizontal=horiz,
        ))

    # --- Junctions ---
    junc_list = getattr(arch, "_junctions", [])
    for junc in junc_list:
        px, py = junc.pos
        jlabel = getattr(junc, "label", f"J{junc.idx}")
        junctions.append(_JunctionInfo(px, py, jlabel))

        # Also capture ions stored in junctions (temporarily during transport)
        for ion in getattr(junc, "ions", []):
            ion_positions[ion.idx] = ion.pos

    # --- Crossings → edges ---
    crossing_list = getattr(arch, "_crossings", [])
    seen_pairs: set = set()
    for crossing in crossing_list:
        conn = getattr(crossing, "connection", (None, None))
        if len(conn) < 2 or conn[0] is None or conn[1] is None:
            continue
        src, tgt = conn
        pair = (min(src.idx, tgt.idx), max(src.idx, tgt.idx))
        if pair in seen_pairs:
            continue
        seen_pairs.add(pair)
        sx, sy = src.pos
        tx, ty = tgt.pos
        edges.append(_EdgeInfo(
            sx, sy, tx, ty,
            label=f"E({src.idx}\u2194{tgt.idx})",
        ))

        # Capture crossing ions
        c_ion = getattr(crossing, "ion", None)
        if c_ion is not None:
            ion_positions[c_ion.idx] = c_ion.pos

    # --- Info ---
    n_ions = len(ion_positions)
    info_lines = [
        f"Traps: {len(traps)}   Junctions: {len(junctions)}",
        f"Ions: {n_ions}",
    ]
    arch_name = type(arch).__name__
    auto_title = f"{arch_name}  [{n_ions} ions]"

    return _GraphLayout(
        traps=traps, junctions=junctions, edges=edges,
        ion_positions=ion_positions, info_lines=info_lines,
        auto_title=auto_title,
    )


# =============================================================================
# Position reading helpers — authoritative ion position extraction
# =============================================================================

def read_ion_positions(arch: Any) -> Dict[int, Tuple[float, float]]:
    """Read current ion positions from arch internals (post-refreshGraph).

    Returns a dict ``{ion_idx: (x, y)}`` covering all ions in traps,
    junctions, and crossings.  This is the single source of truth for
    where ions are after an ``op.run()`` + ``arch.refreshGraph()`` cycle.

    Ion positions are computed **relative to their parent container**
    using the viz spacing constants (``SPACING * ION_SPACING_RATIO``),
    not the raw architecture coordinates (``Trap._spacing = 10``).
    This guarantees ions always render inside their trap rectangles.

    For WISE architectures, coordinates are mapped to the synthetic WISE
    layout space (used by ``_draw_wise_grid``) via ``_wise_normalizer``.
    """
    wise = _is_wise_arch(arch)
    if wise:
        _sx, _sy = _wise_normalizer(arch)
        def _norm(pos):
            return (float(pos[0]) * _sx, float(pos[1]) * _sy)
    else:
        scale = _detect_arch_scale(arch)
        def _norm(pos):
            return (float(pos[0]) / scale, float(pos[1]) / scale)

    viz_ion_spacing = SPACING * ION_SPACING_RATIO

    positions: Dict[int, Tuple[float, float]] = {}

    # Ions in manipulation traps — position by slot index within trap
    for trap in getattr(arch, "_manipulationTraps", []):
        if not hasattr(trap, "pos"):
            continue
        ions_list = list(getattr(trap, "ions", []))
        horiz = getattr(trap, "isHorizontal",
                        getattr(trap, "_isHorizontal", True))
        # Sort by raw position so ions maintain spatial ordering
        # (e.g., ion entering from right stays on right after merge)
        if horiz:
            ions_list.sort(
                key=lambda ion: getattr(ion, '_positionX', 0))
        else:
            ions_list.sort(
                key=lambda ion: getattr(ion, '_positionY', 0))
        n = len(ions_list)
        cx, cy = _norm(trap.pos)
        for i, ion in enumerate(ions_list):
            offset = (i - (n - 1) / 2.0) * viz_ion_spacing
            if horiz:
                positions[ion.idx] = (cx + offset, cy)
            else:
                positions[ion.idx] = (cx, cy + offset)

    # Ions in junctions — place at junction centre
    for junc in getattr(arch, "_junctions", []):
        jx, jy = _norm(junc.pos)
        junc_ions = list(getattr(junc, "ions", []))
        n = len(junc_ions)
        for i, ion in enumerate(junc_ions):
            # Slight spread so overlapping ions are visible
            offset = (i - (n - 1) / 2.0) * viz_ion_spacing * 0.5
            positions[ion.idx] = (jx + offset, jy)

    # Ions in crossings — show on the correct side so that
    # merge / split animations interpolate without visual swaps.
    _CROSS_SIDE_FRAC = 0.35      # how far from midpoint toward the node
    for crossing in getattr(arch, "_crossings", []):
        c_ion = getattr(crossing, "ion", None)
        if c_ion is not None:
            conn = getattr(crossing, "connection", (None, None))
            if len(conn) >= 2 and conn[0] is not None and conn[1] is not None:
                src, tgt = conn
                sx, sy = _norm(src.pos)
                tx, ty = _norm(tgt.pos)
                mx, my = (sx + tx) / 2.0, (sy + ty) / 2.0
                # Shift toward the node the ion is near
                at_src = getattr(crossing, '_ionAtSource', None)
                if at_src is True:
                    positions[c_ion.idx] = (
                        mx + (sx - mx) * _CROSS_SIDE_FRAC,
                        my + (sy - my) * _CROSS_SIDE_FRAC,
                    )
                elif at_src is False:
                    positions[c_ion.idx] = (
                        mx + (tx - mx) * _CROSS_SIDE_FRAC,
                        my + (ty - my) * _CROSS_SIDE_FRAC,
                    )
                else:
                    positions[c_ion.idx] = (mx, my)
            else:
                raw = (getattr(c_ion, "_positionX", c_ion.pos[0]),
                       getattr(c_ion, "_positionY", c_ion.pos[1]))
                positions[c_ion.idx] = _norm(raw)

    return positions


def read_trap_geometries(arch: Any) -> Dict[str, list]:
    """Read static architecture geometry for custom grid drawing.

    Returns ``{'traps': [...], 'junctions': [...], 'crossings': [...]}``.
    Each entry is a dict with position, size, and metadata fields.
    
    Coordinates are normalized to match the viz geometry constants.
    For WISE architectures, coordinates are mapped to the synthetic
    WISE layout space via ``_wise_normalizer``.
    """
    wise = _is_wise_arch(arch)
    if wise:
        _sx, _sy = _wise_normalizer(arch)
        def _norm(pos):
            return (float(pos[0]) * _sx, float(pos[1]) * _sy)
    else:
        scale = _detect_arch_scale(arch)
        def _norm(pos):
            return (float(pos[0]) / scale, float(pos[1]) / scale)
    
    traps: List[dict] = []
    for t in getattr(arch, "_manipulationTraps", []):
        if not hasattr(t, "pos"):
            continue
        ni = len(list(getattr(t, "ions", [])))
        horiz = getattr(t, "isHorizontal",
                        getattr(t, "_isHorizontal", True))
        # Use our SPACING constant for sizing, not the old arch's spacing
        spacing = SPACING * ION_SPACING_RATIO
        inner = max(0, (ni - 1)) * spacing
        if horiz:
            tw = max(1.6, inner + 2 * TRAP_PAD_X)
            th = max(0.7, 2 * ION_RADIUS + 2 * TRAP_PAD_Y)
        else:
            tw = max(0.7, 2 * ION_RADIUS + 2 * TRAP_PAD_Y)
            th = max(1.6, inner + 2 * TRAP_PAD_X)
        
        # Normalize position
        norm_pos = _norm(t.pos)
        
        traps.append({
            "idx": t.idx,
            "pos": norm_pos,
            "width": tw,
            "height": th,
            "n_ions": ni,
            "is_horizontal": horiz,
            "label": getattr(t, "label",
                             getattr(t, "_label", f"T{t.idx}")),
        })

    junctions: List[dict] = []
    for j in getattr(arch, "_junctions", []):
        norm_pos = _norm(j.pos)
        junctions.append({
            "idx": j.idx,
            "pos": norm_pos,
            "n_ions": len(list(getattr(j, "ions", []))),
        })

    crossings: List[dict] = []
    seen: set = set()
    for c in getattr(arch, "_crossings", []):
        conn = getattr(c, "connection", (None, None))
        if len(conn) < 2 or conn[0] is None or conn[1] is None:
            continue
        src, tgt = conn
        pair = (min(src.idx, tgt.idx), max(src.idx, tgt.idx))
        if pair in seen:
            continue
        seen.add(pair)
        # Normalize positions
        src_norm = _norm(src.pos)
        tgt_norm = _norm(tgt.pos)
        crossings.append({
            "src_pos": src_norm,
            "tgt_pos": tgt_norm,
            "has_ion": getattr(c, "ion", None) is not None,
        })

    return {"traps": traps, "junctions": junctions, "crossings": crossings}
