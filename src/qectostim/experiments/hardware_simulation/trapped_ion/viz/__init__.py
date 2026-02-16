"""
Visualization subpackage for trapped-ion QCCD simulation.

Provides matplotlib-based rendering of:
- QCCD architecture topology (traps, junctions, crossings, ion positions)
- Step-by-step transport animation with smooth interpolation & laser beams
- Stim timeslice SVG rendering & scrolling sidebar
- WISE reconfiguration / swap-schedule phase visualization
- MP4 / inline video rendering

Usage
-----
>>> from qectostim.experiments.hardware_simulation.trapped_ion.viz import (
...     display_architecture,
...     animate_transport,
...     render_animation,
... )
>>> fig, ax = display_architecture(arch, title="My QCCD grid")
>>> plt.show()
"""
from .display import display_architecture
from .animation import animate_transport
from .render import render_animation
from .reconfig import visualize_reconfiguration, visualize_routing_result
from .layout import read_ion_positions, read_trap_geometries
from .stim_panel import (
    parse_stim_timeslice_svg,
    draw_stim_svg,
    render_stim_timeslice,
    parse_stim_for_sidebar,
    draw_sidebar,
    draw_progress_bar,
    draw_ops_sidebar,
    SidebarEntry,
)

__all__ = [
    "display_architecture",
    "animate_transport",
    "render_animation",
    "visualize_reconfiguration",
    "visualize_routing_result",
    "read_ion_positions",
    "read_trap_geometries",
    "parse_stim_timeslice_svg",
    "draw_stim_svg",
    "render_stim_timeslice",
    "parse_stim_for_sidebar",
    "draw_sidebar",
    "draw_progress_bar",
    "draw_ops_sidebar",
    "SidebarEntry",
]
