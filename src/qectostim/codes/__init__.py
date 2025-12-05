"""QECToStim Codes Module

This module provides the abstract base classes and concrete implementations
for quantum error-correcting codes.

Class Hierarchy:
    Code (ABC) - Base for any quantum code
    ├── StabilizerCode - Stabilizer codes with symplectic representation
    │   ├── CSSCode - CSS codes with separate X/Z check matrices
    │   │   └── TopologicalCSSCode - CSS codes with geometry
    │   └── SubsystemCode - Codes with gauge operators
    └── HomologicalCode - Codes defined via chain complexes
        └── TopologicalCode - Codes with topological structure
"""

# Abstract base classes
from .abstract_code import Code, StabilizerCode, SubsystemCode, PauliString, Coord, CellEmbedding
from .abstract_css import CSSCode, TopologicalCSSCode
from .abstract_homological import HomologicalCode, TopologicalCode

# Code discovery utilities
from .discovery import (
    discover_all_codes,
    get_code_classes,
    get_css_codes,
    get_non_css_codes,
    get_small_test_codes,
    print_code_catalog,
)

# Re-export all concrete implementations from base
from .base import *

__all__ = [
    # Abstract classes
    "Code",
    "StabilizerCode", 
    "SubsystemCode",
    "CSSCode",
    "TopologicalCSSCode",
    "HomologicalCode",
    "TopologicalCode",
    "PauliString",
    "Coord",
    "CellEmbedding",
    
    # Discovery utilities
    "discover_all_codes",
    "get_code_classes",
    "get_css_codes",
    "get_non_css_codes",
    "get_small_test_codes",
    "print_code_catalog",
]