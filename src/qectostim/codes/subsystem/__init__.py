"""
Subsystem code family.

Subsystem codes have gauge degrees of freedom in addition to logical qubits.
Gauge operators form a non-abelian group, and stabilizers are products of
gauge operators that commute with everything.

Bacon-Shor Codes:
    BaconShorCode: Parameterized Bacon-Shor subsystem code on Lx × Ly grid

Subsystem Surface Codes:
    SubsystemSurfaceCode: Subsystem version of surface code
    SubsystemSurface3, SubsystemSurface5: Pre-configured at d=3, d=5

Gauge Color Codes:
    GaugeColorCode: Subsystem version of color code
    GaugeColor3: Pre-configured gauge color code

Gauge Fixed Codes:
    GaugeFixedCode: Convert subsystem codes to stabilizer codes by fixing gauge
"""

from .bacon_shor import BaconShorCode
from .subsystem_codes import (
    SubsystemSurfaceCode,
    GaugeColorCode,
    SubsystemSurface3,
    SubsystemSurface5,
    GaugeColor3,
)

__all__ = [
    "BaconShorCode",
    "SubsystemSurfaceCode",
    "GaugeColorCode",
    "SubsystemSurface3",
    "SubsystemSurface5",
    "GaugeColor3",
]
