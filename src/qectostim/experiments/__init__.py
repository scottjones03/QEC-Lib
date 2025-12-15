"""QECToStim Experiments Module

This module provides experiment classes for simulating quantum error correction.

Class Hierarchy:
    Experiment (ABC) - Base class for all experiments
    └── MemoryExperiment (ABC) - Repeated stabilizer measurements
        └── StabilizerMemoryExperiment - For general stabilizer codes
            └── CSSMemoryExperiment - Optimized for CSS codes
                └── ColorCodeMemoryExperiment - For color codes with Chromobius support
                └── XYZColorCodeMemoryExperiment - For XYZ color codes with C_XYZ gates
    
    FaultTolerantGadgetExperiment - Memory-Gadget-Memory pattern for FT gates
    LogicalGateExperiment - Automatic gadget routing for logical gates
"""

from .experiment import Experiment
from .memory import (
    MemoryExperiment,
    StabilizerMemoryExperiment,
    CSSMemoryExperiment,
    ColorCodeMemoryExperiment,
    XYZColorCodeMemoryExperiment,
)
from .concatenated_memory import ConcatenatedMemoryExperiment
from .ft_gadget_experiment import (
    FaultTolerantGadgetExperiment,
    FTGadgetExperimentResult,
    run_ft_gadget_experiment,
)
from .logical_gates import (
    LogicalGateExperiment,
    GateRouter,
    GateRoute,
    get_gate_route,
    run_logical_gate,
)
from .stabilizer_rounds import (
    DetectorContext,
    StabilizerRoundBuilder,
    GeneralStabilizerRoundBuilder,
    XYZColorCodeStabilizerRoundBuilder,
    StabilizerBasis,
    get_logical_support,
)

__all__ = [
    # Base
    "Experiment",
    # Memory experiments
    "MemoryExperiment",
    "StabilizerMemoryExperiment",
    "CSSMemoryExperiment",
    "ColorCodeMemoryExperiment",
    "XYZColorCodeMemoryExperiment",
    "ConcatenatedMemoryExperiment",
    # Gadget experiments
    "FaultTolerantGadgetExperiment",
    "FTGadgetExperimentResult",
    "run_ft_gadget_experiment",
    # Logical gate experiments
    "LogicalGateExperiment",
    "GateRouter",
    "GateRoute",
    "get_gate_route",
    "run_logical_gate",
    # Utilities
    "DetectorContext",
    "StabilizerRoundBuilder",
    "GeneralStabilizerRoundBuilder",
    "XYZColorCodeStabilizerRoundBuilder",
    "StabilizerBasis",
    "get_logical_support",
]