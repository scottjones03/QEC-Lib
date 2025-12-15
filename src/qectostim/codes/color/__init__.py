"""
Color code family.

This module contains implementations of color codes - CSS codes defined on
D-colorable lattices where stabilizers are associated with colored plaquettes.

2D Color Codes:
    TriangularColourCode: Color code on triangular (6.6.6) lattice
    HexagonalColourCode: Color code on hexagonal lattice variant
    ColourCode488: Color code on 4.8.8 (square-octagon) tiling
    TruncatedTrihexColorCode: Color code on 4.6.12 tiling

3D Color Codes:
    ColorCode3D: 3D color code on tetrahedral lattice
    ColorCode3DPrism: Stacked 2D color codes with coupling
    TetrahedralColorCode: 3D color code on tetrahedra
    CubicHoneycombColorCode: 3D bitruncated cubic honeycomb
    BallColorCode: Color codes on D-dimensional balls/hyperoctahedra

Hyperbolic Color Codes:
    HyperbolicColorCode: Color codes on hyperbolic tilings
"""

# Import from local files
from .triangular_colour import (
    TriangularColourCode,
    TriangularColour3,
    TriangularColour5, 
    TriangularColour7,
)
from .triangular_colour_xyz import (
    TriangularColourCodeXYZ,
    TriangularColourXYZ3,
    TriangularColourXYZ5,
    TriangularColourXYZ7,
)
from .hexagonal_colour import HexagonalColourCode
from .colour_code import ColourCode488

# New 3D color codes
from .color_3d import (
    ColorCode3D,
    ColorCode3DPrism,
    ColorCode3D_d3,
    ColorCode3D_d5,
    ColorCode3DPrism_2x3,
)

# Extended color codes
from .extended_color import (
    TruncatedTrihexColorCode,
    HyperbolicColorCode,
    BallColorCode,
    CubicHoneycombColorCode,
    TetrahedralColorCode,
    TruncatedTrihex_2x2,
    HyperbolicColor_45_g2,
    HyperbolicColor_64_g2,
    BallColor_3D,
    BallColor_4D,
    CubicHoneycomb_L2,
    Tetrahedral_L2,
)

# Pin and rainbow codes
from .pin_rainbow_codes import (
    QuantumPinCode,
    DoublePinCode,
    RainbowCode,
    HolographicRainbowCode,
    QuantumPin_d3_m2,
    QuantumPin_d5_m3,
    DoublePin_d3,
    DoublePin_d5,
    Rainbow_L3_r3,
    Rainbow_L5_r4,
    HolographicRainbow_L4_d2,
    HolographicRainbow_L6_d3,
)

__all__ = [
    # 2D
    "TriangularColourCode",
    "TriangularColour3",
    "TriangularColour5",
    "TriangularColour7",
    "TriangularColourCodeXYZ",
    "TriangularColourXYZ3",
    "TriangularColourXYZ5",
    "TriangularColourXYZ7",
    "HexagonalColourCode",
    "ColourCode488",
    "TruncatedTrihexColorCode",
    "TruncatedTrihex_2x2",
    # 3D
    "ColorCode3D",
    "ColorCode3DPrism",
    "ColorCode3D_d3",
    "ColorCode3D_d5",
    "ColorCode3DPrism_2x3",
    "TetrahedralColorCode",
    "Tetrahedral_L2",
    "CubicHoneycombColorCode",
    "CubicHoneycomb_L2",
    "BallColorCode",
    "BallColor_3D",
    "BallColor_4D",
    # Hyperbolic
    "HyperbolicColorCode",
    "HyperbolicColor_45_g2",
    "HyperbolicColor_64_g2",
    # Pin codes
    "QuantumPinCode",
    "DoublePinCode",
    "QuantumPin_d3_m2",
    "QuantumPin_d5_m3",
    "DoublePin_d3",
    "DoublePin_d5",
    # Rainbow codes
    "RainbowCode",
    "HolographicRainbowCode",
    "Rainbow_L3_r3",
    "Rainbow_L5_r4",
    "HolographicRainbow_L4_d2",
    "HolographicRainbow_L6_d3",
]
