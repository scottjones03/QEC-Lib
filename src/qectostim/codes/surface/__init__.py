"""
Surface and Toric code family.

This module contains implementations of surface codes, toric codes, and related
topological codes defined on 2D/3D/4D lattices with local stabilizers.

Standard 2D Surface Codes:
    RotatedSurfaceCode: Distance-d rotated planar surface code
    XZZXSurfaceCode: XZZX variant with improved noise bias resilience
    XZZXSurface3, XZZXSurface5: Pre-configured XZZX codes at d=3, d=5
    KitaevSurfaceCode: Generic surface code on arbitrary 2D cellulation

Toric Codes (2D):
    ToricCode: Parameterized toric code on Lx × Ly torus
    ToricCode33: Fixed 3×3 toric code
    TwistedToricCode: Toric code with twisted boundary conditions
    ProjectivePlaneSurfaceCode: Surface code on RP² (non-orientable)

3D Toric Codes:
    ToricCode3D: 3D toric code with qubits on edges
    ToricCode3DFaces: 3D toric code with qubits on faces

4D Toric Codes:
    ToricCode4D: 4D toric code with qubits on 2-cells (self-correcting)
    LoopToricCode4D: (2,2) 4D toric code with loop excitations

Hyperbolic Surface Codes:
    HyperbolicSurfaceCode: General {p,q} tessellation on genus-g surface
    Hyperbolic45Code, Hyperbolic57Code, Hyperbolic38Code: Pre-configured

Exotic Surface Codes:
    FractalSurfaceCode: Sierpinski-carpet fractal geometry
    LCSCode: Lift-connected surface code (stacked with LP couplings)
"""

# Now local imports
from .rotated_surface import RotatedSurfaceCode
from .xzzx_surface import XZZXSurfaceCode, XZZXSurface3, XZZXSurface5
from .toric_code import ToricCode33
from .toric_code_general import ToricCode

# Higher-dimensional toric codes
from .toric_3d import ToricCode3D, ToricCode3DFaces, ToricCode3D_3x3x3, ToricCode3D_4x4x4
from .toric_4d import ToricCode4D, ToricCode4D_2, ToricCode4D_3

# Hyperbolic surface codes
from .hyperbolic import (
    HyperbolicSurfaceCode,
    Hyperbolic45Code,
    Hyperbolic57Code,
    Hyperbolic38Code,
    Hyperbolic45_G2,
    Hyperbolic57_G2,
    Hyperbolic38_G2,
    FreedmanMeyerLuoCode,
    GuthLubotzkyCode,
    GoldenCode,
    FreedmanMeyerLuo_4,
    FreedmanMeyerLuo_5,
    GuthLubotzky_4,
    GuthLubotzky_5,
    GoldenCode_5,
    GoldenCode_8,
)

# Exotic surface codes
from .exotic_surface import (
    FractalSurfaceCode,
    TwistedToricCode,
    ProjectivePlaneSurfaceCode,
    KitaevSurfaceCode,
    LCSCode,
    LoopToricCode4D,
    FractalSurface_L2,
    FractalSurface_L3,
    TwistedToric_4x4,
    ProjectivePlane_4,
    LCS_3x3,
    LoopToric4D_2,
)

__all__ = [
    # Standard 2D
    "RotatedSurfaceCode",
    "XZZXSurfaceCode",
    "XZZXSurface3",
    "XZZXSurface5",
    "KitaevSurfaceCode",
    # Toric 2D
    "ToricCode",
    "ToricCode33",
    "TwistedToricCode",
    "ProjectivePlaneSurfaceCode",
    "TwistedToric_4x4",
    "ProjectivePlane_4",
    # Toric 3D
    "ToricCode3D",
    "ToricCode3DFaces",
    "ToricCode3D_3x3x3",
    "ToricCode3D_4x4x4",
    # Toric 4D
    "ToricCode4D",
    "ToricCode4D_2",
    "ToricCode4D_3",
    "LoopToricCode4D",
    "LoopToric4D_2",
    # Hyperbolic
    "HyperbolicSurfaceCode",
    "Hyperbolic45Code",
    "Hyperbolic57Code",
    "Hyperbolic38Code",
    "Hyperbolic45_G2",
    "Hyperbolic57_G2",
    "Hyperbolic38_G2",
    "FreedmanMeyerLuoCode",
    "GuthLubotzkyCode",
    "GoldenCode",
    "FreedmanMeyerLuo_4",
    "FreedmanMeyerLuo_5",
    "GuthLubotzky_4",
    "GuthLubotzky_5",
    "GoldenCode_5",
    "GoldenCode_8",
    # Exotic
    "FractalSurfaceCode",
    "FractalSurface_L2",
    "FractalSurface_L3",
    "LCSCode",
    "LCS_3x3",
]
