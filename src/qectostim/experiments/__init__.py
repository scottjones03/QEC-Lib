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
    
    FlatConcatenatedMemoryExperiment - Flat model of concatenated codes
        (treats concatenated code as single large stabilizer code)
    
    UnifiedHierarchicalMemoryV2 - Proper hierarchical concatenation (V2)
        (outer operations use FT logical gadgets on inner code blocks)
        
Hierarchical Concatenation Components (V2 Architecture):
========================================================

CRITICAL ARCHITECTURAL INSIGHT: External Outer Syndrome Computation
The outer code syndrome is computed EXTERNALLY by HierarchicalV3Decoder,
NOT tracked through stim's DEM system. Circuits contain ONLY inner detectors.

    LogicalBlockManagerV2 - Manages inner code blocks (emits INNER detectors only)
    OuterStabilizerEngineV2 - FT outer stabilizer measurement (NO outer detectors)
    LogicalGateDispatcher - Selects and emits logical gate implementations

The V1 OuterStabilizerEngine is DEPRECATED - use V2 with emit_outer_detectors=False.
"""

from .experiment import Experiment
from .memory import (
    MemoryExperiment,
    StabilizerMemoryExperiment,
    CSSMemoryExperiment,
    ColorCodeMemoryExperiment,
    XYZColorCodeMemoryExperiment,
    FloquetMemoryExperiment,
)
# Commented out missing module - concatenated_memory.py not found
# from .concatenated_memory import (
#     FlatConcatenatedMemoryExperiment,
#     ConcatenatedMemoryExperiment,  # Deprecated alias
# )

# Concatenated Memory Experiments
from .concatenated_memory import (
    ConcatenatedCSSMemoryExperiment,
    FlatConcatenatedMemoryExperiment,
    HierarchicalConcatenatedMemoryExperiment,
    ConcatenatedMemoryExperiment,  # Deprecated alias
)

# V2 Hierarchical Architecture - try to import if available
try:
    from .unified_hierarchical_memory_v2 import (
        UnifiedHierarchicalMemoryV2,
        HierarchicalMemoryMetadataV2,
    )
except (ImportError, ModuleNotFoundError):
    UnifiedHierarchicalMemoryV2 = None
    HierarchicalMemoryMetadataV2 = None

try:
    from .logical_block_manager_v2 import (
        LogicalBlockManagerV2,
        BlockInfoV2,
        BlockType,
    )
except (ImportError, ModuleNotFoundError):
    LogicalBlockManagerV2 = None
    BlockInfoV2 = None
    BlockType = None

try:
    from .outer_stabilizer_engine_v3 import (
        OuterStabilizerEngineV3,  # V3 - Current implementation with entanglement tracking
        OuterStabInfo,
        OuterStabSet,
        OuterMeasResult,
    )
except (ImportError, ModuleNotFoundError):
    OuterStabilizerEngineV3 = None
    OuterStabInfo = None
    OuterStabSet = None
    OuterMeasResult = None

try:
    from .logical_gate_dispatcher import (
        LogicalGateDispatcher,
        GateMethod,
        GateType,
        GateEmissionResult,
    )
except (ImportError, ModuleNotFoundError):
    LogicalGateDispatcher = None
    GateMethod = None
    GateType = None
    GateEmissionResult = None

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
    FlatConcatenatedStabilizerRoundBuilder,
    HierarchicalConcatenatedStabilizerRoundBuilder,
    ConcatenatedStabilizerRoundBuilder,  # Deprecated alias
    GeneralStabilizerRoundBuilder,
    XYZColorCodeStabilizerRoundBuilder,
    StabilizerBasis,
    get_logical_support,
)

# Multi-level concatenated memory experiment
try:
    from .multilevel_memory import (
        MultiLevelMemoryExperiment,
        MultiLevelMetadata,
        ECMethod,
        ECConfig,
        run_multilevel_experiment,
    )
except (ImportError, ModuleNotFoundError):
    MultiLevelMemoryExperiment = None
    MultiLevelMetadata = None
    ECMethod = None
    ECConfig = None
    run_multilevel_experiment = None

__all__ = [
    # Base
    "Experiment",
    # Memory experiments
    "MemoryExperiment",
    "StabilizerMemoryExperiment",
    "CSSMemoryExperiment",
    "ColorCodeMemoryExperiment",
    "XYZColorCodeMemoryExperiment",
    "FloquetMemoryExperiment",
    # Concatenated memory experiments
    "ConcatenatedCSSMemoryExperiment",
    "FlatConcatenatedMemoryExperiment",
    "HierarchicalConcatenatedMemoryExperiment",
    "ConcatenatedMemoryExperiment",  # Deprecated alias
    # Multi-level concatenated memory
    "MultiLevelMemoryExperiment",
    "MultiLevelMetadata",
    "ECMethod",
    "ECConfig",
    "run_multilevel_experiment",
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
    # V2 Hierarchical Concatenation
    "UnifiedHierarchicalMemoryV2",
    "HierarchicalMemoryMetadataV2",
    "LogicalBlockManagerV2",
    "BlockInfoV2",
    "BlockType",
    "OuterStabilizerEngineV3",
    "OuterStabInfo",
    "OuterStabSet",
    "OuterMeasResult",
    "LogicalGateDispatcher",
    "GateMethod",
    "GateType",
    "GateEmissionResult",
    # Utilities
    "DetectorContext",
    "StabilizerRoundBuilder",
    "FlatConcatenatedStabilizerRoundBuilder",
    "ConcatenatedStabilizerRoundBuilder",  # Deprecated alias
    "GeneralStabilizerRoundBuilder",
    "XYZColorCodeStabilizerRoundBuilder",
    "StabilizerBasis",
    "get_logical_support",
]