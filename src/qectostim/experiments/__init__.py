"""QECToStim Experiments Module

This module provides experiment classes for simulating quantum error correction.

Class Hierarchy:
    Experiment (ABC) - Base class for all experiments
    └── MemoryExperiment (ABC) - Repeated stabilizer measurements
        └── StabilizerMemoryExperiment - For general stabilizer codes
            └── CSSMemoryExperiment - Optimized for CSS codes
                └── ColorCodeMemoryExperiment - For color codes with Chromobius support
"""

from .experiment import Experiment
from .memory import (
    MemoryExperiment,
    StabilizerMemoryExperiment,
    CSSMemoryExperiment,
    ColorCodeMemoryExperiment,
)

__all__ = [
    "Experiment",
    "MemoryExperiment",
    "StabilizerMemoryExperiment",
    "CSSMemoryExperiment",
    "ColorCodeMemoryExperiment",
]