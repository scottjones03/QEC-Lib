"""
Generic code construction utilities.

This module provides flexible tools for constructing quantum error correction
codes from parity-check matrices (CSS) or symplectic stabilizer matrices (general).

Classes:
    GenericCSSCode: Build CSS codes from Hx and Hz matrices
    GenericStabilizerCode: Build stabilizer codes from symplectic matrices
    QLDPCCode: Base class for quantum LDPC codes (algebraic, no geometry)
    TopologicalCSSCodeFromMatrix: Base for CSS codes with geometric embedding
"""

# Now local imports
from .css_generic import GenericCSSCode
from .generic_stabilizer_code import GenericStabilizerCode
from .qldpc_base import QLDPCCode
from .topological_base import TopologicalCSSCodeFromMatrix

__all__ = [
    "GenericCSSCode",
    "GenericStabilizerCode",
    "QLDPCCode",
    "TopologicalCSSCodeFromMatrix",
]
