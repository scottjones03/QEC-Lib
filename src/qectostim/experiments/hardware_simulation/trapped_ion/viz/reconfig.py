"""
Reconfiguration and routing-result visualisation.

* ``visualize_reconfiguration()`` – grid of sub-plots showing per-phase
  ion layouts of a WISE reconfiguration.
* ``visualize_routing_result()`` – quick overlay of routed ions on the
  architecture diagram.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .constants import (
    DPI,
    FONT_SIZE,
    ION_RADIUS,
    INFO_FONT,
    STROKE_THIN,
    _ROLE_COLORS,
    QUBIT_DEFAULT,
)
from .display import (
    display_architecture,
    _draw_ion,
    _ion_color,
    _ion_role_letter,
    _require_matplotlib,
)

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


def visualize_reconfiguration(
    arch: Any,
    phases: Sequence[Dict[int, Tuple[float, float]]],
    fig: Any = None,
    title: str = "WISE Reconfiguration",
    ion_roles: Optional[Dict[int, str]] = None,
) -> Any:
    """Render phases of a WISE reconfiguration as a grid of sub-plots.

    Parameters
    ----------
    arch : QCCDArch | QCCDWiseArch
        Architecture instance (used only for metadata).
    phases : list[dict[int, tuple]]
        Each phase maps ``ion_idx -> (x, y)`` positions.
    fig : Figure | None
        Existing figure to reuse (optional).
    title : str
    ion_roles : dict | None

    Returns
    -------
    matplotlib.figure.Figure
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
    fig, axes = plt.subplots(
        nrows, cols,
        figsize=(5.5 * cols, 4.5 * nrows),
        dpi=DPI,
        squeeze=False,
    )
    axes_flat = axes.flat

    for idx, phase in enumerate(phases):
        axp = axes_flat[idx]
        axp.set_title(f"Phase {idx}", fontsize=FONT_SIZE - 1,
                      fontweight="bold")

        for ion_idx, (ix, iy) in phase.items():
            _draw_ion(axp, ix, iy, ion_idx, set(), ion_roles,
                      show_label=True, radius=ION_RADIUS * 0.75, zorder=5)

        if idx > 0:
            prev = phases[idx - 1]
            for ion_idx, (ix, iy) in phase.items():
                if ion_idx in prev:
                    px, py = prev[ion_idx]
                    if abs(px - ix) > 0.01 or abs(py - iy) > 0.01:
                        axp.annotate(
                            "", xy=(ix, iy), xytext=(px, py),
                            arrowprops=dict(
                                arrowstyle="-|>",
                                color="#42A5F5",
                                lw=1.8, alpha=0.6,
                                connectionstyle="arc3,rad=0.12",
                            ),
                        )
        axp.set_aspect("equal")
        axp.axis("off")

    for idx in range(n_phases, nrows * cols):
        axes_flat[idx].axis("off")

    fig.suptitle(title, fontsize=FONT_SIZE + 2, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    return fig


def visualize_routing_result(
    arch: Any,
    routing_result: Any,
    ion_roles: Optional[Dict[int, str]] = None,
    title: str = "Routing Result",
) -> Tuple[Any, Any]:
    """Quick visualisation of a RoutedCircuit.

    Draws the architecture with highlighted ion pairs that were routed,
    plus a summary of routing operations.

    Parameters
    ----------
    arch : QCCDArch | QCCDWiseArch
    routing_result : RoutedCircuit
    ion_roles : dict | None
    title : str

    Returns
    -------
    (Figure, Axes)
    """
    _require_matplotlib()
    ops = (routing_result.operations
           if hasattr(routing_result, "operations") else [])
    involved: set = set()
    for op in ops:
        if hasattr(op, "qubits"):
            for q in op.qubits:
                involved.add(q)
        elif hasattr(op, "ions"):
            for ion in op.ions:
                idx = getattr(ion, "idx", None)
                if idx is not None:
                    involved.add(idx)

    fig, ax = display_architecture(
        arch,
        title=title,
        highlight_qubits=list(involved),
        ion_roles=ion_roles,
    )
    overhead = getattr(routing_result, "routing_overhead", 0)
    ax.text(
        0.99, 0.01,
        f"Routing ops: {overhead}\nInvolved ions: {len(involved)}",
        transform=ax.transAxes,
        fontsize=INFO_FONT,
        va="bottom", ha="right",
        fontfamily="monospace",
        bbox=dict(
            facecolor="#E8F5E9", alpha=0.9,
            edgecolor="#66BB6A",
            boxstyle="round,pad=0.5",
        ),
        zorder=100,
    )
    return fig, ax


__all__ = ["visualize_reconfiguration", "visualize_routing_result"]
