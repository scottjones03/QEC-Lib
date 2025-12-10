"""
Bosonic and rotor code family.

This module contains implementations of quantum error correction codes
for continuous-variable (bosonic) and rotor systems, defined via
integer-valued homology rather than GF(2).

Integer-Homology Bosonic Codes:
    IntegerHomologyBosonicCode: Bosonic stabilizer codes from integer chain complexes
    HomologicalRotorCode: Rotor CSS codes from chain complexes over Z

Number-Phase Codes:
    HomologicalNumberPhaseCode: Number/phase bosonic codes with torsion logicals

GKP Codes:
    GKPSurfaceCode: GKP encoding combined with surface code structure
"""

from .bosonic_codes import (
    BosonicCode,
    IntegerHomologyBosonicCode,
    HomologicalRotorCode,
    HomologicalNumberPhaseCode,
    GKPSurfaceCode,
    IntegerHomology_L3_2D,
    IntegerHomology_L4_3D,
    RotorCode_L3,
    RotorCode_L5,
    NumberPhase_L3_T2,
    NumberPhase_L4_T3,
    GKPSurface_3x3,
    GKPSurface_5x5,
)

__all__ = [
    # Base class
    "BosonicCode",
    # Integer homology
    "IntegerHomologyBosonicCode",
    "IntegerHomology_L3_2D",
    "IntegerHomology_L4_3D",
    # Rotor codes
    "HomologicalRotorCode",
    "RotorCode_L3",
    "RotorCode_L5",
    # Number-phase codes
    "HomologicalNumberPhaseCode",
    "NumberPhase_L3_T2",
    "NumberPhase_L4_T3",
    # GKP surface
    "GKPSurfaceCode",
    "GKPSurface_3x3",
    "GKPSurface_5x5",
]
