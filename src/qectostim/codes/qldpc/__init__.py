"""
Quantum LDPC (Low-Density Parity-Check) code family.

This module contains implementations of quantum LDPC codes, primarily
based on homological product constructions.

Hypergraph Product Codes:
    HypergraphProductCode: Tillich-Zémor construction from classical codes
    HGPHamming7: HGP of two Hamming [7,4,3] codes

Balanced Product Codes:
    BalancedProductCode: Product modded out by symmetries (better k/n, d/n than HGP)
    DistanceBalancedCode: HGP variant rebalancing X/Z distances

Lifted Product Codes:
    LiftedProductCode: General lifted-product construction
    AbelianLPCode: LP codes for general Abelian groups
    ExpanderLPCode: QLDPC from lifted product of Tanner codes on expanders

Bivariate Bicycle Codes:
    BivariateBicycleCode: BB codes from bivariate polynomials
    BBGrossCode: Gross rate BB code

Generalized Bicycle Codes:
    GeneralizedBicycleCode: GB codes from circulant matrices

Fiber Bundle Codes:
    FiberBundleCode: Codes from fiber bundle structures

Expander-based QLDPC:
    HDXCode: Higher-dimensional expander codes (Ramanujan complexes)
    TensorProductHDXCode: Iterated products of multiple Ramanujan complexes
    DHLVCode: Dinur–Hsieh–Lin–Vidick asymptotically good QLDPC
    DLVCode: Dinur–Lin–Vidick quantum locally testable codes
    LosslessExpanderBPCode: Balanced product of lossless expander graphs

Higher-Dimensional Homological Product:
    HigherDimHomProductCode: Tensor product of ≥2 chain complexes
    SquareHomProductCode: Homological product with square boundary maps
    CampbellDoubleHomProductCode: Double HGP (length-4 complex, single-shot)

Tanner Codes:
    QuantumTannerCode: Original 2D Cayley-complex construction
    GeneralizedQuantumTannerCode: Tanner codes from two commuting regular graphs

Two-Block Group Algebra Codes:
    TwoBlockGroupAlgebraCode: Group-algebra LP codes of length 2|G|
"""

# Import from local files (moved from base/)
from .hypergraph_product import (
    HypergraphProductCode,
    HGPHamming7,
    HGPRep5,
    create_hgp_repetition,
    create_hgp_hamming,
)
from .bivariate_bicycle import (
    BivariateBicycleCode,
    BBGrossCode,
    create_bb_small_12,
    create_bb_tiny,
)
from .lifted_product import LiftedProductCode, GeneralizedBicycleCode
from .fiber_bundle import (
    FiberBundleCode,
    create_fiber_bundle_repetition,
    create_fiber_bundle_hamming,
)

# Import new QLDPC codes from this folder
from .balanced_product import (
    BalancedProductCode,
    DistanceBalancedCode,
    create_balanced_product_repetition,
    create_balanced_product_hamming,
    BalancedProductRep5,
    BalancedProductHamming,
)
from .hdx_codes import (
    HDXCode,
    QuantumTannerCode,
    DinurLinVidickCode,
    HDX_4,
    HDX_6,
    QuantumTanner_4,
    DLV_8,
)
from .expander_codes import (
    ExpanderLPCode,
    DHLVCode,
    CampbellDoubleHGPCode,
    LosslessExpanderBPCode,
    HigherDimHomProductCode,
    ExpanderLP_10_3,
    ExpanderLP_15_4,
    DHLV_5_1,
    DHLV_7_2,
    CampbellDoubleHGP_3,
    CampbellDoubleHGP_5,
    LosslessExpanderBP_8,
    LosslessExpanderBP_12,
    HigherDimHom_3D,
    HigherDimHom_4D,
)

__all__ = [
    # Hypergraph product
    "HypergraphProductCode",
    "HGPHamming7",
    "HGPRep5",
    "create_hgp_repetition",
    "create_hgp_hamming",
    # Bivariate bicycle
    "BivariateBicycleCode",
    "BBGrossCode",
    "create_bb_small_12",
    "create_bb_tiny",
    # Lifted product
    "LiftedProductCode",
    "GeneralizedBicycleCode",
    # Fiber bundle
    "FiberBundleCode",
    "create_fiber_bundle_repetition",
    "create_fiber_bundle_hamming",
    # Balanced product
    "BalancedProductCode",
    "DistanceBalancedCode",
    "create_balanced_product_repetition",
    "create_balanced_product_hamming",
    "BalancedProductRep5",
    "BalancedProductHamming",
    # HDX codes
    "HDXCode",
    "QuantumTannerCode",
    "DinurLinVidickCode",
    "HDX_4",
    "HDX_6",
    "QuantumTanner_4",
    "DLV_8",
    # Expander-based QLDPC
    "ExpanderLPCode",
    "DHLVCode",
    "CampbellDoubleHGPCode",
    "LosslessExpanderBPCode",
    "HigherDimHomProductCode",
    "ExpanderLP_10_3",
    "ExpanderLP_15_4",
    "DHLV_5_1",
    "DHLV_7_2",
    "CampbellDoubleHGP_3",
    "CampbellDoubleHGP_5",
    "LosslessExpanderBP_8",
    "LosslessExpanderBP_12",
    "HigherDimHom_3D",
    "HigherDimHom_4D",
]
