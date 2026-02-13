# src/qectostim/experiments/hardware_simulation/trapped_ion/routing/compilation_pass.py
"""
WISE routing compilation pass — re-export.

Organizational facade for the :class:`WISERoutingPass`, the main
integration point between WISE routing and the core compilation
pipeline.

Classes
-------
WISERoutingPass
    :class:`RoutingPass` implementation that drives WISE SAT routing,
    block-caching, heating accumulation, and mode-snapshot collection.
"""
from __future__ import annotations

from ._core import WISERoutingPass

__all__ = ["WISERoutingPass"]
