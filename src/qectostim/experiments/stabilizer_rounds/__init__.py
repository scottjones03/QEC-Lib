# src/qectostim/experiments/stabilizer_rounds/__init__.py
"""
Stabilizer Round Builders - Modular circuit construction for FT experiments.

This package provides reusable components for building fault-tolerant circuits
with proper detector continuity across memory → gadget → memory phases.

Key components:
- DetectorContext: Tracks measurement indices and stabilizer state across phases
- StabilizerRoundBuilder classes: Emit stabilizer measurement rounds with proper scheduling
- Utility functions: Parse logical operators and compute observable support

Usage:
    from qectostim.experiments.stabilizer_rounds import (
        DetectorContext,
        CSSStabilizerRoundBuilder,
        StabilizerBasis,
    )
    
    ctx = DetectorContext()
    builder = CSSStabilizerRoundBuilder(code, ctx)
    builder.emit_rounds(circuit, num_rounds=5)
"""
from .context import DetectorContext
from .base import StabilizerBasis, BaseStabilizerRoundBuilder
from .css import CSSStabilizerRoundBuilder
from .color import ColorCodeStabilizerRoundBuilder
from .general import GeneralStabilizerRoundBuilder
from .xyz_color import XYZColorCodeStabilizerRoundBuilder
from .concatenated import (
    FlatConcatenatedStabilizerRoundBuilder,
    ConcatenatedStabilizerRoundBuilder,  # Deprecated alias
)
from .hierarchical_concatenated import HierarchicalConcatenatedStabilizerRoundBuilder
from .utils import get_logical_support, _parse_pauli_support
from .memory_emitter import MemoryRoundEmitter, RoundEmissionConfig, create_memory_emitter

# Backwards compatibility alias
StabilizerRoundBuilder = CSSStabilizerRoundBuilder

__all__ = [
    # Core context
    "DetectorContext",
    
    # Enums
    "StabilizerBasis",
    
    # Builder classes
    "BaseStabilizerRoundBuilder",
    "CSSStabilizerRoundBuilder",
    "ColorCodeStabilizerRoundBuilder",
    "GeneralStabilizerRoundBuilder",
    "XYZColorCodeStabilizerRoundBuilder",
    "FlatConcatenatedStabilizerRoundBuilder",
    "HierarchicalConcatenatedStabilizerRoundBuilder",
    
    # Deprecated aliases (kept for backward compatibility)
    "ConcatenatedStabilizerRoundBuilder",
    
    # Utility functions
    "get_logical_support",
    "_parse_pauli_support",
    
    # Memory round emission
    "MemoryRoundEmitter",
    "RoundEmissionConfig",
    "create_memory_emitter",
    
    # Backwards compatibility
    "StabilizerRoundBuilder",
]
