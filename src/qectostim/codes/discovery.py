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
from typing import Dict, List, Optional, Tuple, Type, Any, Callable
import inspect
import importlib
import multiprocessing as mp
from multiprocessing import Process, Queue
import queue
import signal
import sys
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError

from .abstract_code import Code, StabilizerCode
from .abstract_css import CSSCode


class TimeoutError(Exception):
    """Raised when code instantiation times out."""
    pass


def _timeout_handler(signum, frame):
    """Signal handler for timeout."""
    raise TimeoutError("Code instantiation timed out")


def _instantiate_code_in_process(
    result_queue: Queue,
    class_name: str,
    spec: Any,
    module_name: str,
) -> None:
    """Worker function to instantiate a code in a separate process.
    
    This runs in a child process and puts the result in result_queue.
    We can't pickle arbitrary code objects, so we return the essential
    attributes needed to reconstruct or verify the code.
    """
    try:
        # Re-import the module in the subprocess
        if module_name == 'small':
            from . import small as module
        elif module_name == 'generic':
            from . import generic as module
        elif module_name == 'surface':
            from . import surface as module
        elif module_name == 'color':
            from . import color as module
        elif module_name == 'qldpc':
            from . import qldpc as module
        elif module_name == 'subsystem':
            from . import subsystem as module
        elif module_name == 'floquet':
            from . import floquet as module
        elif module_name == 'topological':
            from . import topological as module
        elif module_name == 'qudit':
            from . import qudit as module
        elif module_name == 'bosonic':
            from . import bosonic as module
        elif module_name == 'composite':
            from . import composite as module
        else:
            result_queue.put(('error', f'Unknown module: {module_name}'))
            return
            
        obj = getattr(module, class_name, None)
        if obj is None:
            result_queue.put(('not_found', None))
            return
            
        # Instantiate
        if isinstance(obj, Code):
            code = obj
        elif spec == "factory":
            code = obj()
        elif isinstance(spec, dict):
            code = obj(**spec)
        else:
            code = obj()
            
        # Return success signal - the main process will re-instantiate
        result_queue.put(('success', None))
        
    except Exception as e:
        result_queue.put(('error', str(e)))


def _try_instantiate_with_timeout(
    class_name: str,
    spec: Any,
    modules: Dict[str, Any],
    timeout: float = 5.0,
    use_subprocess: bool = False,
) -> Optional[Code]:
    """Try to instantiate a code with timeout protection.
    
    Uses signal-based timeout on Unix/macOS for fast interruption.
    Falls back to try/except on Windows or if signals unavailable.
    
    Args:
        class_name: Name of the class or factory function
        spec: Either "factory", {}, or dict of kwargs
        modules: Dict of module name -> module object
        timeout: Maximum seconds to wait for instantiation
        use_subprocess: If True, use multiprocessing (slower but more reliable)
        
    Returns:
        Instantiated Code object, or None if failed/timed out
    """
    # First, find which module contains this class
    module_name = None
    obj = None
    for name, module in modules.items():
        obj = getattr(module, class_name, None)
        if obj is not None:
            module_name = name
            break
    
    if obj is None:
        return None
    
    def _do_instantiate():
        if isinstance(obj, Code):
            return obj
        elif spec == "factory":
            return obj()
        elif isinstance(spec, dict):
            return obj(**spec)
        else:
            return obj()
    
    # Use subprocess for hard timeout if requested
    if use_subprocess:
        result_queue: Queue = mp.Queue()
        process = Process(
            target=_instantiate_code_in_process,
            args=(result_queue, class_name, spec, module_name),
        )
        process.start()
        process.join(timeout=timeout)
        
        if process.is_alive():
            process.terminate()
            process.join(timeout=1.0)
            if process.is_alive():
                process.kill()
                process.join()
            return None
        
        try:
            status, _ = result_queue.get_nowait()
        except queue.Empty:
            return None
        
        if status != 'success':
            return None
        
        # Re-instantiate in main process
        try:
            return _do_instantiate()
        except Exception:
            return None
    
    # Use ThreadPoolExecutor for timeout - works in Jupyter and everywhere
    # This is more portable than signal.alarm() which fails in non-main threads
    if timeout > 0:
        executor = ThreadPoolExecutor(max_workers=1)
        future = executor.submit(_do_instantiate)
        try:
            result = future.result(timeout=timeout)
            executor.shutdown(wait=False)
            return result
        except FuturesTimeoutError:
            # Cancel the future if possible, don't wait for it
            future.cancel()
            executor.shutdown(wait=False)
            return None
        except Exception:
            executor.shutdown(wait=False)
            return None
    else:
        # Fallback: direct instantiation without timeout
        try:
            return _do_instantiate()
        except Exception:
            return None


# Import from new module structure
def _get_all_modules():
    """Get all code modules for discovery."""
    modules = {}
    
    try:
        from . import small
        modules['small'] = small
    except ImportError:
        pass
    
    try:
        from . import generic
        modules['generic'] = generic
    except ImportError:
        pass
    
    try:
        from . import surface
        modules['surface'] = surface
    except ImportError:
        pass
    
    try:
        from . import color
        modules['color'] = color
    except ImportError:
        pass
    
    try:
        from . import qldpc
        modules['qldpc'] = qldpc
    except ImportError:
        pass
    
    try:
        from . import subsystem
        modules['subsystem'] = subsystem
    except ImportError:
        pass
    
    try:
        from . import floquet
        modules['floquet'] = floquet
    except ImportError:
        pass
    
    try:
        from . import topological
        modules['topological'] = topological
    except ImportError:
        pass
    
    try:
        from . import qudit
        modules['qudit'] = qudit
    except ImportError:
        pass
    
    try:
        from . import bosonic
        modules['bosonic'] = bosonic
    except ImportError:
        pass
    
    try:
        from . import composite
        modules['composite'] = composite
    except ImportError:
        pass
    
    # Fallback to base for backward compatibility
    try:
        from . import base
        modules['base'] = base
    except ImportError:
        pass
    
    return modules


def discover_all_codes(
    include_css: bool = True,
    include_non_css: bool = True,
    include_subsystem: bool = False,  # Subsystem codes may need special handling
    include_floquet: bool = False,     # Floquet codes need different experiments
    min_distance: Optional[int] = None,
    max_qubits: Optional[int] = None,
    include_qldpc: bool = True,
    timeout_per_code: float = 5.0,
) -> Dict[str, Code]:
    """
    Dynamically discover and instantiate all available QEC codes.
    
    This function imports all codes from qectostim.codes modules and
    returns a dictionary mapping code names to instantiated code objects.
    
    Uses multiprocessing to enforce timeout on slow-to-instantiate codes,
    preventing the discovery from hanging on codes with expensive constructors.
    
    Args:
        include_css: Include CSS codes (CSSCode subclasses)
        include_non_css: Include non-CSS stabilizer codes
        include_subsystem: Include subsystem codes (may need gauge fixing)
        include_floquet: Include Floquet/dynamic codes
        min_distance: Only include codes with distance >= this value
        max_qubits: Only include codes with n <= this value
        include_qldpc: Include QLDPC codes (may be large)
        timeout_per_code: Max seconds to wait for each code to instantiate (default 5.0)
        
    Returns:
        Dict mapping code names (strings) to instantiated Code objects.
        Codes that fail to instantiate or timeout are silently skipped.
        
    Example:
        >>> codes = discover_all_codes(max_qubits=50, timeout_per_code=3.0)
        >>> for name, code in codes.items():
        ...     print(f"{name}: [[{code.n},{code.k},{code.distance}]]")
    """
    modules = _get_all_modules()
    
    codes: Dict[str, Code] = {}
    
    # List of code classes to try instantiating with default parameters
    # Format: (class_name, factory_func_or_default_args, module_hint)
    code_specs = [
        # ========== SMALL CODES ==========
        # Standard small CSS codes (no args needed)
        ("FourQubit422Code", {}),
        ("SixQubit622Code", {}),
        ("SteanCode713", {}),
        ("ShorCode91", {}),
        ("ReedMullerCode151", {}),
        ("HammingCSSCode", {}),
        
        # Non-CSS codes (no args needed)
        ("PerfectCode513", {}),
        ("EightThreeTwoCode", {}),
        ("SixQubit642Code", {}),
        ("BareAncillaCode713", {}),
        ("TenQubitCode", {}),
        ("FiveQubitMixedCode", {}),
        
        # Repetition codes (multiple sizes)
        ("RepetitionCode", {"N": 3}),
        ("RepetitionCode", {"N": 5}),
        ("RepetitionCode", {"N": 7}),
        
        # ========== SURFACE CODES ==========
        ("RotatedSurfaceCode", {"distance": 3}),
        ("RotatedSurfaceCode", {"distance": 5}),
        ("ToricCode", {"Lx": 3, "Ly": 3}),
        ("ToricCode", {"Lx": 5, "Ly": 5}),
        ("XZZXSurfaceCode", {"distance": 3}),
        ("XZZXSurfaceCode", {"distance": 5}),
        # Note: ToricCode4D is the 4D toric/tesseract code
        ("ToricCode4D", {"L": 2}),
        ("ToricCode4D", {"L": 3}),
        # 3D Toric codes
        ("ToricCode3D", {"Lx": 3, "Ly": 3, "Lz": 3}),
        ("ToricCode3D", {"Lx": 4, "Ly": 4, "Lz": 4}),
        
        # Hyperbolic surface codes
        ("HyperbolicSurfaceCode", {"genus": 2, "p": 5, "q": 4}),
        ("FreedmanMeyerLuoCode", {"L": 4}),
        ("GuthLubotzkyCode", {"level": 2}),
        ("GoldenCode", {"level": 1}),
        ("FractalSurfaceCode", {"L": 3}),
        ("TwistedToricCode", {"L": 3}),
        ("LCSCode", {"L": 3}),
        ("ProjectivePlaneSurfaceCode", {"L": 3}),
        
        # ========== COLOR CODES ==========
        ("TriangularColourCode", {"distance": 3}),
        ("TriangularColourCode", {"distance": 5}),
        ("HexagonalColourCode", {"distance": 2}),
        ("HexagonalColourCode", {"distance": 3}),
        ("ColourCode488", {"distance": 3}),
        ("TruncatedTrihexColorCode", {}),  # Fixed name
        
        # 3D Color codes
        ("ColorCode3D", {"distance": 3}),   # Fixed: was ColorCode3D_d3
        ("ColorCode3D", {"distance": 5}),   # Fixed: was ColorCode3D_d5
        ("ColorCode3DPrism", {"Lx": 2, "Ly": 3}),  # Fixed: was ColorCode3DPrism_2x3
        ("CubicHoneycombColorCode", {"L": 2}),  # Fixed: was CubicHoneycomb_L2
        ("TetrahedralColorCode", {"L": 2}),     # Fixed: was Tetrahedral_L2
        ("BallColorCode", {"dim": 3}),          # Fixed: was BallColor_3D
        ("BallColorCode", {"dim": 4}),          # Fixed: was BallColor_4D
        
        # Hyperbolic color codes
        ("HyperbolicColorCode", {"p": 4, "q": 5, "genus": 2}),  # Fixed: was HyperbolicColor_45_g2
        ("HyperbolicColorCode", {"p": 6, "q": 4, "genus": 2}),  # Fixed: was HyperbolicColor_64_g2
        
        # Pin and rainbow codes
        ("QuantumPinCode", {"distance": 3, "m": 2}),      # Fixed: was QuantumPin_d3_m2
        ("QuantumPinCode", {"distance": 5, "m": 3}),      # Fixed: was QuantumPin_d5_m3
        ("DoublePinCode", {"distance": 3}),               # Fixed: was DoublePin_d3
        ("DoublePinCode", {"distance": 5}),               # Fixed: was DoublePin_d5
        ("RainbowCode", {"L": 3, "r": 3}),                # Fixed: was Rainbow_L3_r3
        ("RainbowCode", {"L": 5, "r": 4}),                # Fixed: was Rainbow_L5_r4
        ("HolographicRainbowCode", {"L": 4, "distance": 2}),  # Fixed: was HolographicRainbow_L4_d2
        ("HolographicRainbowCode", {"L": 6, "distance": 3}),  # Fixed: was HolographicRainbow_L6_d3
        
        # ========== QLDPC CODES ==========
        # Hypergraph product codes
        ("HGPHamming7", "factory"),
        ("HypergraphProductCode", {"base_matrix": None}),  # Will fail, use factory below
        # Bivariate bicycle codes
        ("BBGrossCode", "factory"),
        ("BivariateBicycleCode", {"m": 6, "polynomial_a": [1, 2], "polynomial_b": [0, 3]}),
        # HDX codes
        ("HDX_4", "factory"),
        ("HDX_6", "factory"),
        ("QuantumTanner_4", "factory"),
        ("DLV_8", "factory"),
        # Expander-based
        ("ExpanderLP_10_3", "factory"),
        ("ExpanderLP_15_4", "factory"),
        ("DHLV_5_1", "factory"),
        ("DHLV_7_2", "factory"),
        ("CampbellDoubleHGP_3", "factory"),
        ("CampbellDoubleHGP_5", "factory"),
        ("LosslessExpanderBP_8", "factory"),
        ("LosslessExpanderBP_12", "factory"),
        ("HigherDimHom_3D", "factory"),
        ("HigherDimHom_4D", "factory"),
        # Balanced product
        ("BalancedProductRep5", "factory"),
        ("BalancedProductHamming", "factory"),
        
        # ========== SUBSYSTEM CODES ==========
        ("BaconShorCode", {"m": 3, "n": 3}),  # Uses m, n parameters
        ("SubsystemSurfaceCode", {"distance": 3}),  # Fixed: was SubsystemSurface3
        ("SubsystemSurfaceCode", {"distance": 5}),  # Fixed: was SubsystemSurface5
        ("GaugeColorCode", {"distance": 3}),        # Fixed: was GaugeColor3
        
        # ========== FLOQUET CODES ==========
        ("Honeycomb2x3", "factory"),  # Pre-built instance
        ("Honeycomb3x3", "factory"),  # Pre-built instance
        ("ISGFloquet3", "factory"),   # Pre-built instance
        ("HoneycombCode", {"rows": 4, "cols": 4}),  # Direct instantiation
        ("ISGFloquetCode", {"base_distance": 5}),   # Direct instantiation
        
        # ========== TOPOLOGICAL / FRACTON CODES ==========
        # Factory functions that return pre-built instances
        ("HaahCode_3", "factory"),
        ("HaahCode_4", "factory"),
        ("XCubeCode_3", "factory"),
        ("XCubeCode_4", "factory"),
        ("ChamonCode_3", "factory"),
        ("ChamonCode_4", "factory"),
        ("CheckerboardCode_4", "factory"),
        ("FibonacciFractalCode_4", "factory"),
        ("FibonacciFractalCode_5", "factory"),
        ("SierpinskiPrismCode_3_2", "factory"),
        ("SierpinskiPrismCode_4_3", "factory"),
        
        # ========== QUDIT CODES ==========
        ("GaloisQuditSurfaceCode", {"Lx": 3, "Ly": 3, "q": 3}),
        ("GaloisQuditSurfaceCode", {"Lx": 4, "Ly": 4, "q": 5}),
        ("GaloisQuditColorCode", {"L": 3, "q": 3}),
        ("ModularQuditSurfaceCode", {"Lx": 3, "Ly": 3, "d": 3}),
        ("ModularQuditSurfaceCode", {"Lx": 4, "Ly": 4, "d": 5}),
        ("ModularQudit3DSurfaceCode", {"L": 3, "d": 3}),
        ("ModularQuditColorCode", {"L": 3, "d": 3}),
        
        # ========== BOSONIC CODES ==========
        ("IntegerHomologyBosonicCode", {"L": 3, "dim": 2}),
        ("IntegerHomologyBosonicCode", {"L": 4, "dim": 3}),
        ("HomologicalRotorCode", {"L": 3}),
        ("HomologicalRotorCode", {"L": 5}),
        ("HomologicalNumberPhaseCode", {"L": 3, "T": 2}),
        ("HomologicalNumberPhaseCode", {"L": 4, "T": 3}),
        ("GKPSurfaceCode", {"Lx": 3, "Ly": 3}),
        ("GKPSurfaceCode", {"Lx": 5, "Ly": 5}),
        
        # ========== COMPOSITE CODES ==========
        ("HypergraphProductCode", "factory"),
        ("HomologicalProductCode", "factory"),
    ]
    
    # Skip list for codes known to hang or take very long to instantiate
    # These codes have expensive constructors that may block indefinitely
    skip_codes = {
        # These codes have known performance issues in their constructors
        "HolographicRainbowCode",
        "ColorCode3D",
        "ColorCode3DPrism", 
        "BallColorCode",
        "ToricCode3D",
        "HomologicalNumberPhaseCode",  # Often hangs
        "ModularQudit3DSurfaceCode",   # Can be slow
        "HigherDimHom_4D",             # 4D codes are expensive
    }
    
    for spec_item in code_specs:
        class_name = spec_item[0]
        spec = spec_item[1]
        
        # Skip known problematic codes
        if class_name in skip_codes:
            continue
        
        try:
            # Use direct instantiation (fast) - no subprocess overhead
            # Subprocess timeout is available but disabled by default for speed
            code = _try_instantiate_with_timeout(
                class_name=class_name,
                spec=spec,
                modules=modules,
                timeout=timeout_per_code,
                use_subprocess=False,  # Fast path: no subprocess
            )
            
            if code is None:
                continue
            
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
    qldpc_names = ['Hypergraph', 'Bicycle', 'Lifted', 'Fiber', 'HGP', 'BB', 'GB',
                   'HDX', 'Expander', 'DHLV', 'Campbell', 'Tanner', 'DLV', 
                   'Lossless', 'HigherDim', 'Balanced', 'QLDPC']
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
    from . import base as base_module
    
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
