"""
Gadget Interface and Implementations for Concatenated Code Experiments.

ARCHITECTURE:
=============
This module follows a combinator pattern with clean separation of concerns:

1. GATE IMPLEMENTATION STRATEGIES (the "how gates work"):
   - TransversalStrategy: Bitwise parallel gates (default for CSS codes)
   - TeleportationStrategy: Gate teleportation (placeholder for future)
   - LatticeSurgeryStrategy: Lattice surgery operations (placeholder for future)

2. EC GADGETS (syndrome extraction methods):
   
   StabilizerSyndromeGadget:
       Standard stabilizer measurement with 1 ancilla per stabilizer generator.
       Measures each stabilizer separately (not transversal).
       
   TransversalSyndromeGadget:
       Transversal syndrome extraction (Steane-style) with n ancillas per block.
       Uses transversal CNOTs for fault-tolerant syndrome extraction.
       - Z syndrome: |0⟩^n ancilla + CNOT(data→ancilla) + measure Z
       - X syndrome: |+⟩^n ancilla + CNOT(data→ancilla) + measure X
       
   TeleportationECGadget:
       Teleportation-based EC (Knill's protocol) with 2n ancillas per block.
       Data is teleported through a Bell pair; original data is destroyed.
       Requires Pauli frame tracking in the decoder.

3. META-GADGETS / COMBINATORS (composition patterns):
   - ParallelGadget: Apply a gadget to multiple blocks simultaneously
   - RecursiveGadget: Apply gadgets hierarchically (children-first)
   - ChainedGadget: Apply multiple gadgets in sequence
   - RepeatedGadget: Repeat a gadget multiple times
   - ConditionalGadget: Conditionally apply based on syndrome

4. UTILITIES:
   - GadgetBuilder: Fluent API for complex compositions
   - HierarchicalSyndromeLayout: Track syndrome positions for decoder
   - make_recursive_ec: Convenience function for recursive EC
   - make_parallel_steane_ec: Convenience function for parallel Steane

DEPRECATED NAMES:
=================
The following names are deprecated but kept for backward compatibility:
- SteaneECGadget → Use TransversalSyndromeGadget
- KnillECGadget → Use TransversalSyndromeGadget (NOT teleportation!)
- TrueKnillECGadget → Use TeleportationECGadget

USAGE EXAMPLES:
===============
# Transversal syndrome extraction (recommended for most uses)
from qectostim.experiments.gadgets import TransversalSyndromeGadget
gadget = TransversalSyndromeGadget(code)

# Per-stabilizer syndrome extraction (1 ancilla per stabilizer)
from qectostim.experiments.gadgets import StabilizerSyndromeGadget
gadget = StabilizerSyndromeGadget(code)

# Teleportation-based EC (data destroyed, output on new qubits)
from qectostim.experiments.gadgets import TeleportationECGadget
gadget = TeleportationECGadget(code)

# Parallel: Syndrome extraction on 7 blocks simultaneously  
parallel = ParallelGadget(TransversalSyndromeGadget(inner_code), n_blocks=7)

# Recursive: Different EC at each concatenation level
recursive_ec = RecursiveGadget(
    code=concatenated_code,
    level_gadgets={
        0: TransversalSyndromeGadget(outer_code),
        1: TransversalSyndromeGadget(inner_code),
    }
)

# Repeated: 3 rounds for temporal averaging
repeated = RepeatedGadget(TransversalSyndromeGadget(code), n_rounds=3)
"""

# Base infrastructure
from .base import Gadget, MeasurementMap, SyndromeSchedule, LogicalMeasurementMap, CompositeGadget

# Simple utility gadgets
from .noop_gadget import NoOpGadget, IdleGadget

# Main EC method gadgets (RECOMMENDED)
from .transversal_syndrome_gadget import TransversalSyndromeGadget
from .teleportation_ec_gadget import TeleportationECGadget
from .stabilizer_syndrome_gadget import StabilizerSyndromeGadget
from .shor_syndrome_gadget import ShorSyndromeGadget, decode_shor_syndrome

# Ancilla preparation gadgets (fault-tolerance levels)
from .ancilla_prep_gadget import (
    AncillaPrepMethod,
    AncillaBasis,
    AncillaPrepGadget,
    BareAncillaGadget,
    EncodedAncillaGadget,
    VerifiedAncillaGadget,
    create_ancilla_prep_gadget,
)

# Post-selection utilities
from . import post_selection_utils

from .teleportation_ec_gadget import TrueKnillECGadget

# FinalMeasurementGadget removed - module doesn't exist
# from .steane_ec_gadget import FinalMeasurementGadget

# Meta-gadgets / Combinators (the "how")
from .combinators import (
    # Gate implementation strategies
    GateImplementationStrategy,
    GateStrategyType,
    TransversalStrategy,
    TeleportationStrategy,
    LatticeSurgeryStrategy,
    # Composition combinators
    ParallelGadget,
    ChainedGadget,
    RecursiveGadget,
    ConditionalGadget,
    RepeatedGadget,
    GadgetBuilder,
    HierarchicalSyndromeLayout,
    make_recursive_ec,
    make_parallel_steane_ec,
    # Flag-based FT gadgets
    FlagQubitConfig,
    FlaggedSyndromeGadget,
)

# FT gadget adapter (bridges gadgets/ and experiments/gadgets/)
from .ft_gadget_adapter import FTGadgetAdapter, adapt_ft_gadget, ScheduledECGadget

# NEW: Detector-based architecture
from .detector_primitives import (
    DetectorMap,
    DetectorGroup,
    DetectorType,
    DetectorCounter,
    GadgetResult,
    CircuitMetadata,
    ECRoundResult,
)
from .detector_gadgets import (
    DetectorTransversalSyndrome,
    DetectorNoOp,
)

__all__ = [
    # Base infrastructure
    'Gadget',
    'MeasurementMap', 
    'SyndromeSchedule',
    'LogicalMeasurementMap',
    'CompositeGadget',
    
    # Utility gadgets
    'NoOpGadget',
    'IdleGadget',
    
    # Main EC method gadgets (RECOMMENDED)
    'TransversalSyndromeGadget',
    'TeleportationECGadget',
    'StabilizerSyndromeGadget',
    'ShorSyndromeGadget',
    'decode_shor_syndrome',
    'FinalMeasurementGadget',
    
    # Ancilla preparation gadgets
    'AncillaPrepMethod',
    'AncillaBasis',
    'AncillaPrepGadget',
    'BareAncillaGadget',
    'EncodedAncillaGadget',
    'VerifiedAncillaGadget',
    'create_ancilla_prep_gadget',
    
    # Deprecated names (backward compatibility)
    'SteaneECGadget',      # Use TransversalSyndromeGadget
    'KnillECGadget',       # Use TransversalSyndromeGadget
    'TrueKnillECGadget',   # Use TeleportationECGadget
    
    # Gate implementation strategies
    'GateImplementationStrategy',
    'GateStrategyType',
    'TransversalStrategy',
    'TeleportationStrategy',
    'LatticeSurgeryStrategy',
    
    # Meta-gadgets / Combinators
    'ParallelGadget',
    'ChainedGadget',
    'RecursiveGadget',
    'ConditionalGadget',
    'RepeatedGadget',
    'GadgetBuilder',
    'HierarchicalSyndromeLayout',
    'make_recursive_ec',
    'make_parallel_steane_ec',
    
    # Flag-based FT gadgets
    'FlagQubitConfig',
    'FlaggedSyndromeGadget',
    
    # FT gadget bridge
    'FTGadgetAdapter',
    'adapt_ft_gadget',
    'ScheduledECGadget',
    
    # Post-selection utilities
    'post_selection_utils',
    
    # NEW: Detector-based architecture
    'DetectorMap',
    'DetectorGroup',
    'DetectorType',
    'DetectorCounter',
    'GadgetResult',
    'CircuitMetadata',
    'ECRoundResult',
    'DetectorTransversalSyndrome',
    'DetectorNoOp',
]
