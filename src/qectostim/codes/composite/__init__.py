# src/qectostim/codes/composite/__init__.py
"""
Composite Code Constructions.

This module provides classes for building new CSS codes from existing ones:

Concatenation
-------------
- ConcatenatedCode: Base class for concatenated codes
- ConcatenatedCSSCode: CSS concatenation with proper stabilizer lifting
- ConcatenatedTopologicalCSSCode: Adds geometric layout for topological codes

Dual Codes
----------
- DualCode: Swap X and Z sectors (transversal Hadamard)
- SelfDualCode: Marker class for self-dual codes
- dual(): Convenience function

Subcodes
--------
- Subcode: Freeze logical qubits by promoting to stabilizers
- PuncturedCode: Remove physical qubits
- ShortenedCode: Fix and remove physical qubits

Gauge Fixing
------------
- GaugeFixedCode: Convert subsystem codes to stabilizer codes
- gauge_fix_css(): Convenience function

Homological Products
--------------------
- HypergraphProductCode: Tillich-Zémor hypergraph product
- HomologicalProductCode: Alias for HypergraphProductCode
- hypergraph_product(): Convenience function
- homological_product(): General homological product

Examples
--------
>>> from qectostim.codes.composite import (
...     ConcatenatedCSSCode,
...     DualCode,
...     HypergraphProductCode,
... )
>>> from qectostim.codes.base import SteaneCode, RepetitionCode
>>> 
>>> # Concatenate Steane with itself
>>> steane = SteaneCode()
>>> concat = ConcatenatedCSSCode(steane, steane)
>>> print(f"Concatenated: [[{concat.n}, {concat.k}]]")
>>>
>>> # Dual of repetition code
>>> rep = RepetitionCode(3)
>>> dual_rep = DualCode(rep)  # Phase-flip code
>>>
>>> # Hypergraph product
>>> product = HypergraphProductCode(rep, rep)
"""

from .concatenated import (
    ConcatenatedCode,
    ConcatenatedCSSCode,
    ConcatenatedTopologicalCSSCode,
)

from .dual import (
    DualCode,
    SelfDualCode,
    dual,
    swap_pauli_type,
)

from .subcode import (
    Subcode,
    PuncturedCode,
    ShortenedCode,
)

from .gauge_fixed import (
    GaugeFixedCode,
    gauge_fix_css,
)

from .homological_product import (
    HypergraphProductCode,
    HomologicalProductCode,
    TillichZemorHGP,
    hypergraph_product,
    homological_product,
    hypergraph_product_from_classical,
)

from .multilevel_concatenated import (
    CodeNode,
    MultiLevelConcatenatedCode,
    ConcatenatedCodeBuilder,
    build_steane_tower,
    build_standard_concatenation,
)


__all__ = [
    # Concatenation
    'ConcatenatedCode',
    'ConcatenatedCSSCode',
    'ConcatenatedTopologicalCSSCode',
    # Dual
    'DualCode',
    'SelfDualCode',
    'dual',
    'swap_pauli_type',
    # Subcode
    'Subcode',
    'PuncturedCode',
    'ShortenedCode',
    # Gauge fixing
    'GaugeFixedCode',
    'gauge_fix_css',
    # Homological products
    'HypergraphProductCode',
    'HomologicalProductCode',
    'TillichZemorHGP',
    'hypergraph_product',
    'homological_product',
    'hypergraph_product_from_classical',
    # Multi-level concatenation
    'CodeNode',
    'MultiLevelConcatenatedCode',
    'ConcatenatedCodeBuilder',
    'build_steane_tower',
    'build_standard_concatenation',
]
