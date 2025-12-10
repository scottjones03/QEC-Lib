"""
Qudit code family.

This module contains implementations of quantum error correction codes
generalized to qudit (d-level) systems, including both modular qudits
and Galois-field qudits.

Galois-Qudit Codes:
    GaloisQuditSurfaceCode: 2D surface codes generalized to Galois qudits
    GaloisQuditHGPCode: HGP over q-ary classical inputs
    GaloisQuditExpanderCode: Hypergraph-product codes from expander codes over F_q
    GaloisQuditColorCode: Color codes generalized to Galois qudits

Modular-Qudit Codes:
    ModularQuditSurfaceCode: Surface code generalized to modular Z_d qudits
    ModularQudit3DSurfaceCode: 3D surface code on Z_d qudits
    ModularQuditColorCode: Color codes on modular qudits
"""

from .galois import (
    GaloisQuditCode,
    GaloisQuditSurfaceCode,
    GaloisQuditHGPCode,
    GaloisQuditColorCode,
    GaloisQuditExpanderCode,
    GaloisSurface_3x3_GF3,
    GaloisSurface_4x4_GF5,
    GaloisHGP_GF3_n5,
    GaloisHGP_GF5_n7,
    GaloisColor_L3_GF3,
    GaloisExpander_n8_GF3,
)

from .modular import (
    ModularQuditCode,
    ModularQuditSurfaceCode,
    ModularQudit3DSurfaceCode,
    ModularQuditColorCode,
    ModularSurface_3x3_d3,
    ModularSurface_4x4_d5,
    ModularSurface3D_L3_d3,
    ModularSurface3D_L4_d5,
    ModularColor_L3_d3,
    ModularColor_L4_d5,
)

__all__ = [
    # Galois-qudit base
    "GaloisQuditCode",
    # Galois-qudit surface
    "GaloisQuditSurfaceCode",
    "GaloisSurface_3x3_GF3",
    "GaloisSurface_4x4_GF5",
    # Galois-qudit HGP
    "GaloisQuditHGPCode",
    "GaloisHGP_GF3_n5",
    "GaloisHGP_GF5_n7",
    # Galois-qudit color
    "GaloisQuditColorCode",
    "GaloisColor_L3_GF3",
    # Galois-qudit expander
    "GaloisQuditExpanderCode",
    "GaloisExpander_n8_GF3",
    # Modular-qudit base
    "ModularQuditCode",
    # Modular-qudit surface
    "ModularQuditSurfaceCode",
    "ModularSurface_3x3_d3",
    "ModularSurface_4x4_d5",
    # Modular-qudit 3D surface
    "ModularQudit3DSurfaceCode",
    "ModularSurface3D_L3_d3",
    "ModularSurface3D_L4_d5",
    # Modular-qudit color
    "ModularQuditColorCode",
    "ModularColor_L3_d3",
    "ModularColor_L4_d5",
]
