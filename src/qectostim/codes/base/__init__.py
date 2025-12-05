"""Base CSS codes for QECToStim.

This module contains a comprehensive library of QEC codes including:
- Standard CSS codes (Steane, Shor, Reed-Muller)
- Topological codes (Surface, Toric, Colour)
- QLDPC codes (Hypergraph Product, Bivariate Bicycle, Lifted Product)
- Subsystem codes (Bacon-Shor, Gauge Color)
- Non-CSS codes (Perfect, [[8,3,2]])
- Floquet codes (Honeycomb)
"""

# Original CSS codes
from .four_two_two import FourQubit422Code
from .six_two_two import SixQubit622Code
from .steane_713 import SteanCode713
from .shor_code import ShorCode91
from .reed_muller_code import ReedMullerCode151
from .toric_code import ToricCode33
from .repetition_codes import RepetitionCode, create_repetition_code_3, create_repetition_code_5, create_repetition_code_7, create_repetition_code_9
from .css_generic import GenericCSSCode
from .rotated_surface import RotatedSurfaceCode

# === New QEC Codes ===

# Non-CSS stabilizer codes
from .perfect_code import PerfectCode513
from .eight_three_two import EightThreeTwoCode
from .non_css_codes import (
    SixQubit642Code, BareAncillaCode713, TenQubitCode, FiveQubitMixedCode,
    NonCSS642, NonCSS713, NonCSS1023, Mixed512
)

# Generic stabilizer code (for building non-CSS codes from matrices)
from .generic_stabilizer_code import GenericStabilizerCode

# General Toric Code (parameterized)
from .toric_code_general import ToricCode

# QLDPC codes
from .hypergraph_product import HypergraphProductCode, create_hgp_repetition, create_hgp_hamming, HGPHamming7
from .bivariate_bicycle import BivariateBicycleCode, create_bb_gross_code, create_bb_small_12, create_bb_tiny, BBGrossCode
from .lifted_product import LiftedProductCode, create_lifted_product_repetition, GeneralizedBicycleCode, create_gb_15_code, create_gb_21_code
from .fiber_bundle import FiberBundleCode, create_fiber_bundle_repetition, create_fiber_bundle_hamming

# Colour codes
from .triangular_colour import TriangularColourCode
from .hexagonal_colour import HexagonalColourCode
from .colour_code import ColourCode488

# Subsystem codes
from .bacon_shor import BaconShorCode
from .subsystem_codes import SubsystemSurfaceCode, GaugeColorCode, SubsystemSurface3, SubsystemSurface5, GaugeColor3

# Surface code variants
from .xzzx_surface import XZZXSurfaceCode, XZZXSurface3, XZZXSurface5

# Floquet codes
from .floquet_codes import HoneycombCode, ISGFloquetCode, Honeycomb2x3, Honeycomb3x3, ISGFloquet3

# Hamming-based CSS codes
from .hamming_css import HammingCSSCode

__all__ = [
    # Original codes
    "FourQubit422Code",
    "SixQubit622Code",
    "SteanCode713",
    "ShorCode91",
    "ReedMullerCode151",
    "ToricCode33",
    "RepetitionCode",
    "create_repetition_code_3",
    "create_repetition_code_5",
    "create_repetition_code_7",
    "create_repetition_code_9",
    "GenericCSSCode",
    "RotatedSurfaceCode",
    
    # Non-CSS codes
    "PerfectCode513",
    "EightThreeTwoCode",
    "SixQubit642Code",
    "BareAncillaCode713",
    "TenQubitCode",
    "FiveQubitMixedCode",
    "NonCSS642",
    "NonCSS713",
    "NonCSS1023",
    "Mixed512",
    
    # Generic codes
    "GenericStabilizerCode",
    
    # Toric (parameterized)
    "ToricCode",
    
    # QLDPC codes
    "HypergraphProductCode",
    "create_hgp_repetition",
    "create_hgp_hamming",
    "HGPHamming7",
    "BivariateBicycleCode",
    "create_bb_gross_code",
    "create_bb_small_12",
    "create_bb_tiny",
    "BBGrossCode",
    "LiftedProductCode",
    "create_lifted_product_repetition",
    "GeneralizedBicycleCode",
    "create_gb_15_code",
    "create_gb_21_code",
    "FiberBundleCode",
    "create_fiber_bundle_repetition",
    "create_fiber_bundle_hamming",
    
    # Colour codes
    "TriangularColourCode",
    "HexagonalColourCode",
    "ColourCode488",
    
    # Subsystem codes
    "BaconShorCode",
    "SubsystemSurfaceCode",
    "SubsystemSurface3",
    "SubsystemSurface5",
    "GaugeColorCode",
    "GaugeColor3",
    
    # Surface code variants
    "XZZXSurfaceCode",
    "XZZXSurface3",
    "XZZXSurface5",
    
    # Floquet codes
    "HoneycombCode",
    "ISGFloquetCode",
    "Honeycomb2x3",
    "Honeycomb3x3",
    "ISGFloquet3",
    
    # Hamming CSS
    "HammingCSSCode",
]
