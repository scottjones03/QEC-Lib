"""Code discovery utilities for QECToStim.

This module provides utilities to dynamically discover and instantiate
all available QEC codes in the codebase. This is useful for:
- Running comprehensive tests across all codes
- Building code catalogs
- Checking coverage of decoder support

Main function:
- discover_all_codes(): Returns dict mapping code names to instantiated codes
"""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple, Type, Any
import inspect
import importlib

from qectostim.codes.abstract_code import Code, StabilizerCode
from qectostim.codes.abstract_css import CSSCode


def discover_all_codes(
    include_css: bool = True,
    include_non_css: bool = True,
    include_subsystem: bool = False,  # Subsystem codes may need special handling
    include_floquet: bool = False,     # Floquet codes need different experiments
    min_distance: Optional[int] = None,
    max_qubits: Optional[int] = None,
    include_qldpc: bool = True,
) -> Dict[str, Code]:
    """
    Dynamically discover and instantiate all available QEC codes.
    
    This function imports all codes from qectostim.codes.base and
    returns a dictionary mapping code names to instantiated code objects.
    
    Args:
        include_css: Include CSS codes (CSSCode subclasses)
        include_non_css: Include non-CSS stabilizer codes
        include_subsystem: Include subsystem codes (may need gauge fixing)
        include_floquet: Include Floquet/dynamic codes
        min_distance: Only include codes with distance >= this value
        max_qubits: Only include codes with n <= this value
        include_qldpc: Include QLDPC codes (may be large)
        
    Returns:
        Dict mapping code names (strings) to instantiated Code objects.
        Codes that fail to instantiate are silently skipped.
        
    Example:
        >>> codes = discover_all_codes(max_qubits=50)
        >>> for name, code in codes.items():
        ...     print(f"{name}: [[{code.n},{code.k},{code.distance}]]")
    """
    from qectostim.codes import base as base_module
    
    codes: Dict[str, Code] = {}
    
    # List of code classes to try instantiating with default parameters
    # Format: (class_name, factory_func_or_default_args)
    code_specs = [
        # Standard small CSS codes (no args needed)
        ("FourQubit422Code", {}),
        ("SixQubit622Code", {}),
        ("SteanCode713", {}),
        ("ShorCode91", {}),
        ("ReedMullerCode151", {}),
        ("ToricCode33", {}),
        ("HammingCSSCode", {}),
        
        # Non-CSS codes (no args needed)
        ("PerfectCode513", {}),
        ("EightThreeTwoCode", {}),
        ("SixQubit642Code", {}),
        ("BareAncillaCode713", {}),
        ("TenQubitCode", {}),
        ("FiveQubitMixedCode", {}),
        
        # Topological codes with size parameters
        ("RotatedSurfaceCode", {"distance": 3}),
        ("RotatedSurfaceCode", {"distance": 5}),  # Also test d=5
        ("ToricCode", {"lx": 3, "ly": 3}),
        ("ToricCode", {"lx": 5, "ly": 5}),  # Also test larger toric
        ("TriangularColourCode", {"distance": 3}),
        ("HexagonalColourCode", {"distance": 3}),
        ("ColourCode488", {"distance": 3}),
        ("XZZXSurfaceCode", {"distance": 3}),
        ("XZZXSurface3", {}),  # Pre-built d=3 XZZX
        ("XZZXSurface5", {}),  # Pre-built d=5 XZZX
        
        # QLDPC codes - use factory functions or defaults
        ("HGPHamming7", {}),
        ("BBGrossCode", {}),
        ("create_hgp_repetition", "factory"),
        ("create_hgp_hamming", "factory"),
        ("create_bb_small_12", "factory"),
        ("create_bb_tiny", "factory"),
        ("create_gb_15_code", "factory"),
        ("create_gb_21_code", "factory"),
        ("create_fiber_bundle_repetition", "factory"),
        ("create_fiber_bundle_hamming", "factory"),
        
        # Subsystem codes
        ("BaconShorCode", {"lx": 3, "ly": 3}),
        ("SubsystemSurface3", {}),
        ("SubsystemSurface5", {}),
        ("GaugeColor3", {}),
        
        # Floquet codes
        ("Honeycomb2x3", {}),
        ("Honeycomb3x3", {}),
        ("ISGFloquet3", {}),
        
        # Repetition codes (multiple sizes)
        ("RepetitionCode", {"N": 3}),
        ("RepetitionCode", {"N": 5}),
        ("RepetitionCode", {"N": 7}),
        ("RepetitionCode", {"N": 9}),
    ]
    
    for class_name, spec in code_specs:
        try:
            if spec == "factory":
                # It's a factory function
                factory = getattr(base_module, class_name, None)
                if factory is None:
                    continue
                code = factory()
            else:
                # It's a class with constructor args
                cls = getattr(base_module, class_name, None)
                if cls is None:
                    continue
                if isinstance(spec, dict):
                    code = cls(**spec)
                else:
                    code = cls()
            
            # Apply filters
            if not _passes_filters(
                code, 
                include_css, 
                include_non_css,
                include_subsystem,
                include_floquet,
                include_qldpc,
                min_distance,
                max_qubits
            ):
                continue
                
            # Generate a clean name
            name = _get_code_name(code, class_name)
            codes[name] = code
            
        except Exception as e:
            # Skip codes that fail to instantiate
            # This is expected for some codes that require specific parameters
            continue
    
    return codes


def _passes_filters(
    code: Code,
    include_css: bool,
    include_non_css: bool,
    include_subsystem: bool,
    include_floquet: bool,
    include_qldpc: bool,
    min_distance: Optional[int],
    max_qubits: Optional[int],
) -> bool:
    """Check if a code passes all filter criteria."""
    
    # Check code type
    is_css = getattr(code, 'is_css', False)
    
    if is_css and not include_css:
        return False
    if not is_css and not include_non_css:
        return False
    
    # Check for subsystem codes
    is_subsystem = hasattr(code, 'gauge_matrix') or 'Subsystem' in type(code).__name__
    if is_subsystem and not include_subsystem:
        return False
    
    # Check for Floquet codes
    is_floquet = 'Floquet' in type(code).__name__ or 'Honeycomb' in type(code).__name__
    if is_floquet and not include_floquet:
        return False
    
    # Check for QLDPC codes
    qldpc_names = ['Hypergraph', 'Bicycle', 'Lifted', 'Fiber', 'HGP', 'BB', 'GB']
    is_qldpc = any(name in type(code).__name__ for name in qldpc_names)
    if is_qldpc and not include_qldpc:
        return False
    
    # Check distance
    if min_distance is not None:
        dist = getattr(code, 'distance', None)
        if dist is not None and dist < min_distance:
            return False
    
    # Check qubit count
    if max_qubits is not None:
        n = getattr(code, 'n', None)
        if n is not None and n > max_qubits:
            return False
    
    return True


def _get_code_name(code: Code, class_name: str) -> str:
    """Generate a clean name for a code instance."""
    # Try to get name from metadata
    if hasattr(code, '_metadata') and isinstance(code._metadata, dict):
        if 'name' in code._metadata:
            return code._metadata['name']
    
    # Try to get from code attributes
    n = getattr(code, 'n', '?')
    k = getattr(code, 'k', '?')
    d = getattr(code, 'distance', '?')
    
    # Clean up class name
    name = class_name.replace('Code', '').replace('create_', '')
    
    return f"{name}_[[{n},{k},{d}]]"


def get_code_classes() -> Dict[str, Type[Code]]:
    """
    Get all Code subclasses defined in qectostim.codes.base.
    
    Returns:
        Dict mapping class names to class types (not instances).
    """
    from qectostim.codes import base as base_module
    
    classes = {}
    
    for name in dir(base_module):
        obj = getattr(base_module, name)
        if (inspect.isclass(obj) and 
            issubclass(obj, Code) and
            obj is not Code and
            obj is not CSSCode and
            obj is not StabilizerCode):
            classes[name] = obj
    
    return classes


def get_css_codes(max_qubits: Optional[int] = None) -> Dict[str, CSSCode]:
    """
    Get all CSS codes, optionally filtered by size.
    
    Args:
        max_qubits: Maximum number of physical qubits
        
    Returns:
        Dict mapping code names to CSSCode instances
    """
    all_codes = discover_all_codes(
        include_css=True,
        include_non_css=False,
        max_qubits=max_qubits
    )
    return {name: code for name, code in all_codes.items() 
            if isinstance(code, CSSCode)}


def get_non_css_codes() -> Dict[str, StabilizerCode]:
    """
    Get all non-CSS stabilizer codes.
    
    Returns:
        Dict mapping code names to StabilizerCode instances
    """
    all_codes = discover_all_codes(
        include_css=False,
        include_non_css=True,
    )
    return {name: code for name, code in all_codes.items() 
            if isinstance(code, StabilizerCode) and not isinstance(code, CSSCode)}


def get_small_test_codes(max_qubits: int = 20) -> Dict[str, Code]:
    """
    Get a selection of small codes suitable for fast testing.
    
    Args:
        max_qubits: Maximum number of physical qubits (default 20)
        
    Returns:
        Dict mapping code names to Code instances
    """
    return discover_all_codes(
        include_css=True,
        include_non_css=True,
        include_subsystem=False,
        include_floquet=False,
        include_qldpc=False,
        max_qubits=max_qubits,
    )


def print_code_catalog(codes: Optional[Dict[str, Code]] = None) -> None:
    """
    Print a formatted catalog of codes.
    
    Args:
        codes: Dict of codes to print. If None, discovers all codes.
    """
    if codes is None:
        codes = discover_all_codes()
    
    print(f"{'Name':<35} {'Type':<12} {'[[n,k,d]]':<15}")
    print("-" * 65)
    
    for name, code in sorted(codes.items()):
        n = getattr(code, 'n', '?')
        k = getattr(code, 'k', '?')
        d = getattr(code, 'distance', '?')
        
        is_css = getattr(code, 'is_css', False)
        code_type = "CSS" if is_css else "Non-CSS"
        
        print(f"{name:<35} {code_type:<12} [[{n},{k},{d}]]")
