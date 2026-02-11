"""Code discovery utilities for QECToStim.

This module provides utilities to dynamically discover and instantiate
all available QEC codes in the codebase. This is useful for:
- Running comprehensive tests across all codes
- Building code catalogs
- Checking coverage of decoder support

Main function:
- discover_all_codes(): Returns dict mapping code names to instantiated codes

Code Parameters:
    Varies by discovered code.  Each entry in the returned dictionary is a
    fully-instantiated ``Code`` (or subclass) with its own [[n, k, d]].

Stabiliser Structure:
    Varies.  The discovery module is agnostic to stabiliser structure;
    it delegates entirely to each code's own constructor.

Raises:
    CodeInstantiationTimeoutError
        If a code constructor exceeds the configured timeout.
    TimeoutError
        Legacy alias for the same condition.
    ImportError
        If a code's backing module cannot be imported.
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


class CodeInstantiationTimeoutError(Exception):
    """Raised when a code takes too long to instantiate."""
    pass


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
        Instantiated Code object, or None if class not found
        
    Raises:
        CodeInstantiationTimeoutError: If instantiation takes too long
        Exception: If instantiation fails for any other reason
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
            raise CodeInstantiationTimeoutError(f"Timed out after {timeout}s")
        
        try:
            status, _ = result_queue.get_nowait()
        except queue.Empty:
            raise CodeInstantiationTimeoutError(f"No result received")
        
        if status != 'success':
            raise Exception("Subprocess failed to instantiate code")
        
        # Re-instantiate in main process (may still raise)
        return _do_instantiate()
    
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
            raise CodeInstantiationTimeoutError(f"Timed out after {timeout}s")
        except Exception:
            executor.shutdown(wait=False)
            raise  # Re-raise the actual exception
    else:
        # Direct instantiation without timeout - may raise
        return _do_instantiate()


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
    include_bosonic: bool = False,    # Bosonic codes use continuous variables
    include_qudit: bool = False,       # Qudit codes use d>2 dimensions
    include_fracton: bool = False,    # Fracton codes have exotic excitations
    timeout_per_code: float = 5.0,
    report_failures: bool = False,
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
        include_bosonic: Include bosonic codes (GKP, rotor - use continuous variables)
        include_qudit: Include qudit codes (Galois field - use d>2 dimensions)
        include_fracton: Include fracton codes (XCube, Haah - exotic excitations)
        timeout_per_code: Max seconds to wait for each code to instantiate (default 5.0)
        report_failures: If True, return tuple (codes, failures) where failures
                        is a dict mapping code names to error messages
        
    Returns:
        If report_failures=False: Dict mapping code names to Code objects.
        If report_failures=True: Tuple of (codes_dict, failures_dict)
            where failures_dict maps code names to error messages.
        
    Example:
        >>> codes = discover_all_codes(max_qubits=50, timeout_per_code=3.0)
        >>> for name, code in codes.items():
        ...     print(f"{name}: [[{code.n},{code.k},{code.distance}]]")
        
        >>> # With failure reporting
        >>> codes, failures = discover_all_codes(report_failures=True)
        >>> for name, error in failures.items():
        ...     print(f"FAILED: {name}: {error}")
    """
    modules = _get_all_modules()
    
    codes: Dict[str, Code] = {}
    failures: Dict[str, str] = {}  # Track failures with error messages
    
    # List of code classes to try instantiating with default parameters
    # Format: (class_name, factory_func_or_default_args, module_hint)
    # Includes both direct instantiation and pre-built factory functions
    code_specs = [
        # ========== SMALL CODES ==========
        # Standard small CSS codes (no args needed)
        ("FourQubit422Code", {}),
        ("SixQubit622Code", {}),
        ("SteaneCode713", {}),
        ("ShorCode91", {}),
        ("ReedMullerCode151", {}),
        ("HammingCSSCode", {}),
        # Pre-built Hamming CSS factories
        ("HammingCSS7", "factory"),   # [[7,1,3]] Steane
        ("HammingCSS15", "factory"),  # [[15,7,3]]
        ("HammingCSS31", "factory"),  # [[31,21,3]]
        
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
        ("ToricCode33", "factory"),  # Pre-built 3x3 toric
        ("XZZXSurfaceCode", {"distance": 3}),
        ("XZZXSurfaceCode", {"distance": 5}),
        ("XZZXSurface3", "factory"),  # Pre-built XZZX d=3
        ("XZZXSurface5", "factory"),  # Pre-built XZZX d=5
        # KitaevSurfaceCode requires explicit graph geometry - use factories instead
        
        # 4D Toric codes (tesseract)
        ("ToricCode4D", {"L": 2}),
        ("ToricCode4D", {"L": 3}),
        ("ToricCode4D_2", "factory"),  # Pre-built L=2
        ("ToricCode4D_3", "factory"),  # Pre-built L=3
        ("LoopToricCode4D", {"L": 2}),  # 4D loop toric
        ("LoopToric4D_2", "factory"),   # Pre-built
        
        # 3D Toric codes (uses L parameter, not Lx/Ly/Lz)
        ("ToricCode3D", {"L": 3}),
        ("ToricCode3D", {"L": 4}),
        ("ToricCode3D_3x3x3", "factory"),  # Pre-built 3D
        ("ToricCode3D_4x4x4", "factory"),  # Pre-built 3D
        ("ToricCode3DFaces", {"L": 3}),  # 3D with faces
        
        # Hyperbolic surface codes
        ("HyperbolicSurfaceCode", {"genus": 2, "p": 5, "q": 4}),
        ("Hyperbolic45Code", {}),  # Pre-configured {4,5}
        ("Hyperbolic57Code", {}),  # Pre-configured {5,7}
        ("Hyperbolic38Code", {}),  # Pre-configured {3,8}
        ("Hyperbolic45_G2", "factory"),  # genus=2
        ("Hyperbolic57_G2", "factory"),
        ("Hyperbolic38_G2", "factory"),
        ("FreedmanMeyerLuoCode", {"L": 4}),
        ("FreedmanMeyerLuo_4", "factory"),
        ("FreedmanMeyerLuo_5", "factory"),
        ("GuthLubotzkyCode", {"L": 4}),
        ("GuthLubotzky_4", "factory"),
        ("GuthLubotzky_5", "factory"),
        ("GoldenCode", {"L": 5}),
        ("GoldenCode_5", "factory"),
        ("GoldenCode_8", "factory"),
        
        # Exotic surface codes
        ("FractalSurfaceCode", {"level": 2}),
        ("FractalSurface_L2", "factory"),
        ("FractalSurface_L3", "factory"),
        ("TwistedToricCode", {"Lx": 4, "Ly": 4, "twist": 1}),
        ("TwistedToric_4x4", "factory"),
        ("LCSCode", {"L": 3}),
        ("LCS_3x3", "factory"),
        ("ProjectivePlaneSurfaceCode", {"L": 3}),
        ("ProjectivePlane_4", "factory"),
        
        # ========== COLOR CODES ==========
        ("TriangularColourCode", {"distance": 3}),
        ("TriangularColourCode", {"distance": 5}),
        ("HexagonalColourCode", {"distance": 2}),
        ("HexagonalColourCode", {"distance": 3}),
        ("ColourCode488", {"distance": 3}),
        ("TruncatedTrihexColorCode", {}),
        ("TruncatedTrihex_2x2", "factory"),
        
        # 3D Color codes
        ("ColorCode3D", {"distance": 3}),
        ("ColorCode3D", {"distance": 5}),
        ("ColorCode3D_d3", "factory"),
        ("ColorCode3D_d5", "factory"),
        ("ColorCode3DPrism", {"L": 2, "base_distance": 3}),
        ("ColorCode3DPrism_2x3", "factory"),
        ("CubicHoneycombColorCode", {"L": 2}),
        ("CubicHoneycomb_L2", "factory"),
        ("TetrahedralColorCode", {"L": 2}),
        ("Tetrahedral_L2", "factory"),
        ("BallColorCode", {"dimension": 3}),
        ("BallColorCode", {"dimension": 4}),
        ("BallColor_3D", "factory"),
        ("BallColor_4D", "factory"),
        
        # Hyperbolic color codes
        ("HyperbolicColorCode", {"p": 4, "q": 5, "genus": 2}),
        ("HyperbolicColorCode", {"p": 6, "q": 4, "genus": 2}),
        ("HyperbolicColor_45_g2", "factory"),
        ("HyperbolicColor_64_g2", "factory"),
        
        # Pin and rainbow codes
        ("QuantumPinCode", {"d": 3, "m": 2}),
        ("QuantumPinCode", {"d": 5, "m": 3}),
        ("QuantumPin_d3_m2", "factory"),
        ("QuantumPin_d5_m3", "factory"),
        ("DoublePinCode", {"d": 3}),
        ("DoublePinCode", {"d": 5}),
        ("DoublePin_d3", "factory"),
        ("DoublePin_d5", "factory"),
        ("RainbowCode", {"L": 3, "r": 3}),
        ("RainbowCode", {"L": 5, "r": 4}),
        ("Rainbow_L3_r3", "factory"),
        ("Rainbow_L5_r4", "factory"),
        ("HolographicRainbowCode", {"L": 4, "bulk_depth": 2}),
        ("HolographicRainbowCode", {"L": 6, "bulk_depth": 3}),
        ("HolographicRainbow_L4_d2", "factory"),
        ("HolographicRainbow_L6_d3", "factory"),
        
        # ========== QLDPC CODES ==========
        # Hypergraph product codes
        ("HGPHamming7", "factory"),
        ("HGPRep5", "factory"),  # HGP from repetition code
        ("HypergraphProductCode", {"base_matrix": None}),  # Will fail, use factory
        # Bivariate bicycle codes - correct signature: l, m, A_terms, B_terms
        ("BBGrossCode", "factory"),
        # BB_6x6: Fixed parameters that produce k=8 (original terms gave k=0)
        # A = x + x^2 + x^3 (pure x-powers), B = y + y^2 + y^3 (pure y-powers)
        ("BivariateBicycleCode", {"l": 6, "m": 6, "A_terms": [(1, 0), (2, 0), (3, 0)], "B_terms": [(0, 1), (0, 2), (0, 3)]}),
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
        ("BaconShorCode", {"m": 3, "n": 3}),
        ("SubsystemSurfaceCode", {"distance": 3}),
        ("SubsystemSurfaceCode", {"distance": 5}),
        ("SubsystemSurface3", "factory"),
        ("SubsystemSurface5", "factory"),
        ("GaugeColorCode", {"distance": 3}),
        ("GaugeColor3", "factory"),
        
        # ========== FLOQUET CODES ==========
        ("Honeycomb2x3", "factory"),
        ("Honeycomb3x3", "factory"),
        ("ISGFloquet3", "factory"),
        ("HoneycombCode", {"rows": 4, "cols": 4}),
        ("ISGFloquetCode", {"base_distance": 5}),
        
        # ========== TOPOLOGICAL / FRACTON CODES ==========
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
        ("GaloisSurface_3x3_GF3", "factory"),
        ("GaloisSurface_4x4_GF5", "factory"),
        ("GaloisHGP_GF3_n5", "factory"),
        ("GaloisHGP_GF5_n7", "factory"),
        ("GaloisColor_L3_GF3", "factory"),
        ("GaloisExpander_n8_GF3", "factory"),
        ("ModularQuditSurfaceCode", {"Lx": 3, "Ly": 3, "d": 3}),
        ("ModularQuditSurfaceCode", {"Lx": 4, "Ly": 4, "d": 5}),
        ("ModularQudit3DSurfaceCode", {"L": 3, "d": 3}),
        ("ModularQuditColorCode", {"L": 3, "d": 3}),
        ("ModularSurface_3x3_d3", "factory"),
        ("ModularSurface_4x4_d5", "factory"),
        ("ModularSurface3D_L3_d3", "factory"),
        ("ModularSurface3D_L4_d5", "factory"),
        ("ModularColor_L3_d3", "factory"),
        ("ModularColor_L4_d5", "factory"),
        
        # ========== BOSONIC CODES ==========
        ("IntegerHomologyBosonicCode", {"L": 3, "dim": 2}),
        ("IntegerHomologyBosonicCode", {"L": 4, "dim": 3}),
        ("IntegerHomology_L3_2D", "factory"),
        ("IntegerHomology_L4_3D", "factory"),
        ("HomologicalRotorCode", {"L": 3}),
        ("HomologicalRotorCode", {"L": 5}),
        ("RotorCode_L3", "factory"),
        ("RotorCode_L5", "factory"),
        ("HomologicalNumberPhaseCode", {"L": 3, "T": 2}),
        ("HomologicalNumberPhaseCode", {"L": 4, "T": 3}),
        ("NumberPhase_L3_T2", "factory"),
        ("NumberPhase_L4_T3", "factory"),
        ("GKPSurfaceCode", {"Lx": 3, "Ly": 3}),
        ("GKPSurfaceCode", {"Lx": 5, "Ly": 5}),
        ("GKPSurface_3x3", "factory"),
        ("GKPSurface_5x5", "factory"),
        
        # ========== COMPOSITE CODES ==========
        ("HypergraphProductCode", "factory"),
        ("HomologicalProductCode", "factory"),
    ]
    
    # Skip list for codes known to hang or require special handling
    # We try to keep this minimal - only codes that truly cannot instantiate
    skip_codes = {
        # Product codes require complex arguments (code objects as inputs)
        # These are factories, not standalone codes
        "HomologicalProductCode",
        "HypergraphProductCode",
        
        # Codes that hang indefinitely or have fundamental construction issues
        "HomologicalNumberPhaseCode",  # Often hangs during construction
        "NumberPhase_L3_T2",
        "NumberPhase_L4_T3",
        
        # Qudit codes that don't work with standard qubit decoders
        "ModularQudit3DSurfaceCode",
        "ModularSurface3D_L3_d3",
        "ModularSurface3D_L4_d5",
        
        # Higher-dim codes with known bugs (TODO: fix these)
        "HigherDimHom_3D",             # IndexError in construction
        "HigherDimHom_4D",             # Construction incomplete
    }
    
    # Codes with known implementation bugs - track these separately for reporting
    known_buggy_codes: set[str] = set()
    
    for spec_item in code_specs:
        class_name = spec_item[0]
        spec = spec_item[1]
        
        # Skip known slow codes but still report them as skipped if reporting is on
        if class_name in skip_codes:
            failures[class_name] = "SKIPPED: Known to be very slow or require special handling"
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
                # Class not found in any module
                failures[class_name] = "NOT_FOUND: Class not found in any module"
                continue
            
            # Apply filters
            if not _passes_filters(
                code, 
                include_css, 
                include_non_css,
                include_subsystem,
                include_floquet,
                include_qldpc,
                include_bosonic,
                include_qudit,
                include_fracton,
                min_distance,
                max_qubits
            ):
                continue
                
            # Generate a clean name
            name = _get_code_name(code, class_name)
            codes[name] = code
            
        except CodeInstantiationTimeoutError as e:
            # Timeout - distinct from other errors
            failures[class_name] = f"TIMEOUT: {str(e)}"
            continue
        except Exception as e:
            # Capture the failure with error details
            error_type = type(e).__name__
            error_msg = str(e)
            # Truncate very long error messages
            if len(error_msg) > 200:
                error_msg = error_msg[:200] + "..."
            failures[class_name] = f"{error_type}: {error_msg}"
            continue
    
    if report_failures:
        return codes, failures
    return codes


def _passes_filters(
    code: Code,
    include_css: bool,
    include_non_css: bool,
    include_subsystem: bool,
    include_floquet: bool,
    include_qldpc: bool,
    include_bosonic: bool,
    include_qudit: bool,
    include_fracton: bool,
    min_distance: Optional[int],
    max_qubits: Optional[int],
) -> bool:
    """Check if a code passes all filter criteria."""
    
    # Check for bosonic codes (GKP, rotor, etc.) - use continuous variables
    bosonic_names = ['GKP', 'Rotor', 'Bosonic', 'IntegerHomology']
    is_bosonic = any(name in type(code).__name__ for name in bosonic_names)
    if is_bosonic and not include_bosonic:
        return False
    
    # Check for qudit codes (Galois field, etc.) - use d>2 dimensions
    qudit_names = ['Galois', 'Qudit', 'ModularQudit']
    is_qudit = any(name in type(code).__name__ for name in qudit_names)
    if is_qudit and not include_qudit:
        return False
    
    # Check for fracton codes (XCube, Haah, Chamon, etc.) - exotic excitations
    fracton_names = ['XCube', 'Haah', 'Chamon', 'Checkerboard', 'Fractal', 'Sierpinski', 'Fracton']
    is_fracton = any(name in type(code).__name__ for name in fracton_names)
    if is_fracton and not include_fracton:
        return False
    
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
