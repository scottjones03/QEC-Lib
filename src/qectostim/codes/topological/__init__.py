"""
Exotic topological and fracton code family.

This module contains implementations of codes with exotic topological properties,
including fracton codes with restricted mobility excitations.

Fracton Codes (Type-I):
    XCubeCode: X-cube model with lineon and planon excitations
    CheckerboardCode: Foliated fracton code on cubic lattice
    ChamonCode: Chamon model on BCC lattice
    XCubeCode_3, XCubeCode_4: Pre-configured X-cube codes
    CheckerboardCode_4: Pre-configured checkerboard code
    ChamonCode_3, ChamonCode_4: Pre-configured Chamon codes

Fracton Codes (Type-II):
    HaahCode: Haah's cubic code with immobile fracton excitations
    HaahCode_3, HaahCode_4: Pre-configured Haah codes

Fractal Codes:
    FibonacciFractalCode: Fractal code on Fibonacci quasilattice
    SierpinskiPrismCode: 3D fractal code on Sierpinski gasket prism
"""

# Fracton codes
from .fracton_codes import (
    XCubeCode,
    HaahCode,
    CheckerboardCode,
    ChamonCode,
    FibonacciFractalCode,
    SierpinskiPrismCode,
    XCubeCode_3,
    XCubeCode_4,
    HaahCode_3,
    HaahCode_4,
    CheckerboardCode_4,
    ChamonCode_3,
    ChamonCode_4,
    FibonacciFractalCode_4,
    FibonacciFractalCode_5,
    SierpinskiPrismCode_3_2,
    SierpinskiPrismCode_4_3,
)

__all__ = [
    # Type-I fracton
    "XCubeCode",
    "CheckerboardCode",
    "ChamonCode",
    "XCubeCode_3",
    "XCubeCode_4",
    "CheckerboardCode_4",
    "ChamonCode_3",
    "ChamonCode_4",
    # Type-II fracton
    "HaahCode",
    "HaahCode_3",
    "HaahCode_4",
    # Fractal codes
    "FibonacciFractalCode",
    "SierpinskiPrismCode",
    "FibonacciFractalCode_4",
    "FibonacciFractalCode_5",
    "SierpinskiPrismCode_3_2",
    "SierpinskiPrismCode_4_3",
]
