"""QECToStim Codes Module

This module provides the abstract base classes and concrete implementations
for quantum error-correcting codes.

Class Hierarchy (Updated):
    Code (ABC) - Base for any quantum code (n, k, logical ops)
    └── StabilizerCode - Stabilizer codes with symplectic representation
        ├── HomologicalCode - Codes defined via chain complexes (inherits StabilizerCode)
        │   └── TopologicalCode - With geometric embedding
        └── SubsystemCode - Codes with gauge operators
    
    CSSCode(StabilizerCode, HomologicalCode) - Diamond inheritance, separate X/Z checks
    ├── TopologicalCSSCode - 3-chain (2D surface/toric), chain_length=3
    ├── TopologicalCSSCode3D - 4-chain (3D toric), chain_length=4  
    └── TopologicalCSSCode4D - 5-chain (4D tesseract), chain_length=5
    
    SubsystemCSSCode(SubsystemCode, CSSCode) - Subsystem codes with CSS structure
    
    FloquetCode(StabilizerCode) - Time-dependent stabilizer codes (NOT CSS)

Chain Complex Structure:
    - 2-chain: C1 → C0 (repetition codes)
    - 3-chain: C2 → C1 → C0 (2D toric/surface codes)
    - 4-chain: C3 → C2 → C1 → C0 (3D toric codes)
    - 5-chain: C4 → C3 → C2 → C1 → C0 (4D tesseract, hypergraph products)

Code Organization:
    codes/
    ├── abstract_*.py      - Base classes
    ├── utils.py           - GF(2) linear algebra utilities
    ├── discovery.py       - Code discovery and cataloging
    ├── homological_product.py - Homological product of chain complexes
    ├── complexes/         - Chain complex infrastructure
    ├── composite/         - Code construction operations
    ├── base/              - Backwards compatibility exports
    ├── generic/           - Generic code builders from matrices
    ├── small/             - Small CSS and non-CSS codes
    ├── surface/           - Surface and toric codes
    ├── color/             - Color codes
    ├── qldpc/             - Quantum LDPC codes
    ├── subsystem/         - Subsystem codes
    ├── floquet/           - Floquet/dynamic codes
    ├── topological/       - Exotic topological codes
    ├── qudit/             - Qudit codes
    └── bosonic/           - Bosonic/rotor codes
"""

# Abstract base classes
from .abstract_code import (
    Code,
    StabilizerCode,
    SubsystemCode,
    PauliString,
    Coord,
    CellEmbedding,
)

from .abstract_css import (
    CSSCode,
    CSSCodeWithComplex,
    TopologicalCSSCode,
    TopologicalCSSCode3D,
    TopologicalCSSCode4D,
    SubsystemCSSCode,
    FloquetCode,
)

from .abstract_homological import HomologicalCode, TopologicalCode

# Chain complexes
from .complexes import (
    ChainComplex,
    CSSChainComplex2,
    CSSChainComplex3,
    CSSChainComplex4,
    FiveCSSChainComplex,
    tensor_product_chain_complex,
)



# Code discovery utilities
from .discovery import (
    discover_all_codes,
    get_code_classes,
    get_css_codes,
    get_non_css_codes,
    get_small_test_codes,
    print_code_catalog,
)

# Re-export from base (for backwards compatibility)
from .base import *

# Composite operations (from the composite folder)
from .composite import (
    ConcatenatedCSSCode,
    ConcatenatedTopologicalCSSCode,
    DualCode, 
    SelfDualCode,
    Subcode, 
    PuncturedCode,
    ShortenedCode,
    GaugeFixedCode,
    HypergraphProductCode as HGPComposite,
    hypergraph_product,
)

__all__ = [
    # Abstract classes
    "Code",
    "StabilizerCode", 
    "SubsystemCode",
    "CSSCode",
    "CSSCodeWithComplex",
    "TopologicalCSSCode",
    "TopologicalCSSCode3D",
    "TopologicalCSSCode4D",
    "SubsystemCSSCode",
    "FloquetCode",
    "HomologicalCode",
    "TopologicalCode",
    "PauliString",
    "Coord",
    "CellEmbedding",
    
    # Chain complexes
    "ChainComplex",
    "CSSChainComplex2",
    "CSSChainComplex3",
    "CSSChainComplex4",
    "FiveCSSChainComplex",
    "tensor_product_chain_complex",

    
    # Discovery utilities
    "discover_all_codes",
    "get_code_classes",
    "get_css_codes",
    "get_non_css_codes",
    "get_small_test_codes",
    "print_code_catalog",
    
    # Composite operations
    "ConcatenatedCSSCode",
    "ConcatenatedTopologicalCSSCode",
    "DualCode",
    "SelfDualCode",
    "Subcode",
    "PuncturedCode",
    "ShortenedCode",
    "GaugeFixedCode",
    "HGPComposite",
    "hypergraph_product",
]
