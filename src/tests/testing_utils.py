# src/tests/testing_utils.py
"""
Shared Testing Utilities for QECToStim.

This module provides reusable testing functions for code validation,
decoder testing, circuit analysis, and visualization. These utilities
are used by example notebooks and can be imported for custom testing.

Usage
-----
>>> from tests.testing_utils import test_decoder_on_code, visualize_code_geometry
>>> result = test_decoder_on_code(code, decoder_class)
>>> visualize_code_geometry(code, ax)
"""
from __future__ import annotations

import sys
import time
import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

import numpy as np

# =============================================================================
# Status Indicators
# =============================================================================

STATUS_OK = "✓"
STATUS_WARN = "⚠️"
STATUS_SKIP = "-"
STATUS_FAIL = "✗"


# =============================================================================
# Module Management
# =============================================================================

def clear_qectostim_modules() -> int:
    """Clear all qectostim modules from sys.modules for fresh imports.
    
    Returns
    -------
    int
        Number of modules cleared.
    """
    modules_to_clear = [m for m in sys.modules if 'qectostim' in m]
    for m in modules_to_clear:
        del sys.modules[m]
    return len(modules_to_clear)


# =============================================================================
# Test Result Structure
# =============================================================================

@dataclass
class TestResult:
    """Result of a decoder or code test."""
    status: str = "UNKNOWN"
    ler: Optional[float] = None
    ler_no_decode: Optional[float] = None
    time_ms: Optional[float] = None
    warnings: List[str] = field(default_factory=list)
    error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for compatibility."""
        return {
            'status': self.status,
            'ler': self.ler,
            'ler_no_decode': self.ler_no_decode,
            'time_ms': self.time_ms,
            'warnings': self.warnings,
            'error': self.error,
        }


# =============================================================================
# Decoder Testing
# =============================================================================

def test_decoder_on_code(
    code,
    decoder_class: Type,
    decoder_name: str = "",
    code_type: str = "CSS",
    p: float = 0.01,
    shots: int = 1000,
    rounds: int = 3,
) -> TestResult:
    """Test a decoder on a code and return results.
    
    Parameters
    ----------
    code : CSSCode or StabilizerCode
        The quantum error-correcting code to test.
    decoder_class : Type
        The decoder class (not instance) to use.
    decoder_name : str
        Name of the decoder for special handling.
    code_type : str
        Type of code: 'CSS', 'Color', 'Non-CSS', 'Subsystem', 'Floquet', 'QLDPC'.
    p : float
        Physical error probability.
    shots : int
        Number of Monte Carlo samples.
    rounds : int
        Number of syndrome measurement rounds.
        
    Returns
    -------
    TestResult
        Test result containing status, LER, timing, and any warnings/errors.
    """
    # Import here to avoid circular imports
    from qectostim.experiments.memory import (
        CSSMemoryExperiment,
        StabilizerMemoryExperiment,
    )
    from qectostim.noise.models import CircuitDepolarizingNoise
    
    # Try to import color code experiment
    try:
        from qectostim.experiments.memory import ColorCodeMemoryExperiment
        has_color_exp = True
    except ImportError:
        has_color_exp = False
        ColorCodeMemoryExperiment = None
    
    # Try to import Chromobius error
    try:
        from qectostim.decoders.chromobius_decoder import ChromobiusIncompatibleError
    except ImportError:
        ChromobiusIncompatibleError = Exception
    
    result = TestResult()
    
    try:
        noise = CircuitDepolarizingNoise(p1=p, p2=p)
        
        # Use appropriate experiment class based on code type
        is_css_code = hasattr(code, 'hx') and hasattr(code, 'hz')
        
        if code_type == 'Color' and has_color_exp:
            exp = ColorCodeMemoryExperiment(code=code, rounds=rounds, noise_model=noise)
        elif is_css_code:
            exp = CSSMemoryExperiment(code=code, rounds=rounds, noise_model=noise)
        else:
            exp = StabilizerMemoryExperiment(code=code, rounds=rounds, noise_model=noise)
        
        circuit = noise.apply(exp.to_stim())
        
        # Try to build DEM - with fallback for decomposition failures
        try:
            dem = circuit.detector_error_model(decompose_errors=True)
        except Exception:
            try:
                dem = circuit.detector_error_model(
                    decompose_errors=True, 
                    ignore_decomposition_failures=True
                )
            except Exception as e2:
                err_msg = str(e2)
                if 'non-deterministic' in err_msg.lower():
                    result.status = 'WARN'
                    result.error = "Mixed logical ops"
                    result.warnings.append('Mixed logical ops')
                else:
                    result.status = 'FAIL'
                    result.error = f"DEM: {err_msg[:30]}"
                return result
        
        # Sample
        sampler = dem.compile_sampler()
        raw = sampler.sample(shots, bit_packed=False)
        
        if isinstance(raw, tuple):
            det_samples = np.asarray(raw[0], dtype=np.uint8)
            obs_samples = np.asarray(raw[1], dtype=np.uint8)
        else:
            arr = np.asarray(raw, dtype=np.uint8)
            det_samples = arr[:, :dem.num_detectors]
            obs_samples = arr[:, dem.num_detectors:]
        
        if obs_samples.shape[1] > 0:
            result.ler_no_decode = float(obs_samples[:, 0].mean())
        
        # Create decoder
        try:
            if decoder_name in ['PyMatching', 'FusionBlossom']:
                decoder = decoder_class(dem)
            else:
                decoder = decoder_class(dem=dem)
        except ChromobiusIncompatibleError:
            result.status = 'SKIP'
            result.error = "Not a color code"
            result.warnings.append('Chromobius: requires color-code DEM')
            return result
        except Exception as e:
            err_msg = str(e)
            if 'chromobius' in err_msg.lower() or 'color' in err_msg.lower():
                result.status = 'SKIP'
                result.error = "Not a color code"
                result.warnings.append('Chromobius: requires color-code DEM')
            else:
                result.status = 'FAIL'
                result.error = f"Decoder init: {err_msg[:25]}"
            return result
        
        # Decode
        try:
            start = time.time()
            corrections = decoder.decode_batch(det_samples)
            result.time_ms = (time.time() - start) * 1000
        except BaseException as e:
            err_msg = str(e)
            err_type = type(e).__name__
            if 'panic' in err_type.lower() or 'matching' in err_msg.lower():
                result.status = 'WARN'
                result.error = f"Decoder issue: {err_type}"
                result.warnings.append('Decoder incompatibility')
            else:
                result.status = 'FAIL'
                result.error = f"Decode: {err_msg[:25]}"
            return result
        
        corrections = np.asarray(corrections, dtype=np.uint8)
        if corrections.ndim == 1:
            corrections = corrections.reshape(-1, max(1, dem.num_observables))
        
        if obs_samples.shape[1] > 0:
            logical_errors = (corrections[:, 0] ^ obs_samples[:, 0]).astype(np.uint8)
            result.ler = float(logical_errors.mean())
        
        result.status = 'OK'
        
        # Add warnings for suspicious results
        if result.ler is not None:
            if result.ler < 1e-6 and p > 0.001:
                result.warnings.append('LER≈0')
            if result.ler_no_decode is not None:
                d = getattr(code, 'metadata', {}).get('distance', 0)
                if result.ler >= result.ler_no_decode and d >= 3:
                    result.warnings.append('No improvement')
        
    except BaseException as e:
        err_type = type(e).__name__
        if 'panic' in err_type.lower():
            result.status = 'WARN'
            result.error = f"Rust panic: {err_type}"
            result.warnings.append('Rust decoder panic')
        else:
            result.status = 'FAIL'
            result.error = str(repr(e))[:30]
    
    return result


# =============================================================================
# Composite Code Testing
# =============================================================================

def test_composite_construction(
    outer_code,
    inner_code,
    composite_class: Type,
    **kwargs,
) -> Tuple[bool, Optional[Any], str]:
    """Test composite code construction.
    
    Parameters
    ----------
    outer_code : CSSCode
        The outer code.
    inner_code : CSSCode
        The inner code.
    composite_class : Type
        The composite code class to test.
    **kwargs
        Additional arguments to pass to the composite class.
        
    Returns
    -------
    Tuple[bool, Optional[CSSCode], str]
        (success, composite_code, message)
    """
    try:
        start = time.time()
        composite = composite_class(outer_code, inner_code, **kwargs)
        elapsed = (time.time() - start) * 1000
        
        # Validate basic properties
        n = composite.n
        k = composite.k
        
        if k <= 0:
            return False, composite, f"Invalid k={k}"
        
        # Verify n is reasonable
        expected_n = outer_code.n * inner_code.n
        
        if hasattr(composite, 'hx') and hasattr(composite, 'hz'):
            # CSS code - check stabilizer matrices
            if composite.hx.shape[1] != n or composite.hz.shape[1] != n:
                return False, composite, "Matrix dimension mismatch"
        
        return True, composite, f"OK (n={n}, k={k}, {elapsed:.1f}ms)"
        
    except Exception as e:
        return False, None, f"{type(e).__name__}: {str(e)[:50]}"


def test_code_circuit(
    code,
    p: float = 0.001,
    rounds: int = 1,
) -> Tuple[bool, Optional[Any], str]:
    """Test that a code can generate a valid Stim circuit.
    
    Parameters
    ----------
    code : CSSCode or StabilizerCode
        The code to test.
    p : float
        Noise probability.
    rounds : int
        Syndrome rounds.
        
    Returns
    -------
    Tuple[bool, stim.Circuit, str]
        (success, circuit, message)
    """
    from qectostim.experiments.memory import CSSMemoryExperiment, StabilizerMemoryExperiment
    from qectostim.noise.models import CircuitDepolarizingNoise
    
    try:
        noise = CircuitDepolarizingNoise(p1=p, p2=p)
        
        if hasattr(code, 'hx') and hasattr(code, 'hz'):
            exp = CSSMemoryExperiment(code=code, rounds=rounds, noise_model=noise)
        else:
            exp = StabilizerMemoryExperiment(code=code, rounds=rounds, noise_model=noise)
        
        circuit = exp.to_stim()
        
        # Basic validation
        n_qubits = circuit.num_qubits
        n_detectors = circuit.num_detectors
        n_observables = circuit.num_observables
        
        if n_qubits < code.n:
            return False, circuit, f"Too few qubits: {n_qubits} < {code.n}"
        if n_detectors == 0:
            return False, circuit, "No detectors"
        if n_observables == 0:
            return False, circuit, "No observables"
        
        return True, circuit, f"OK ({n_qubits}q, {n_detectors}det, {n_observables}obs)"
        
    except Exception as e:
        return False, None, f"{type(e).__name__}: {str(e)[:50]}"


# =============================================================================
# Circuit Analysis
# =============================================================================

@dataclass
class CircuitAnalysis:
    """Analysis results for a Stim circuit."""
    num_qubits: int = 0
    num_detectors: int = 0
    num_observables: int = 0
    circuit_depth: int = 0
    num_ticks: int = 0
    gate_counts: Dict[str, int] = field(default_factory=dict)
    two_qubit_gates: int = 0
    single_qubit_gates: int = 0
    has_qubit_coords: bool = False
    qubit_coords: Optional[Dict[int, List[float]]] = None
    
    # Parallelism metrics
    ops_per_tick: List[int] = field(default_factory=list)
    avg_ops_per_tick: float = 0.0
    max_ops_per_tick: int = 0
    parallelism_score: float = 0.0  # higher = more parallel


def analyze_stim_circuit(circuit) -> CircuitAnalysis:
    """Analyze a Stim circuit for key properties.
    
    Parameters
    ----------
    circuit : stim.Circuit
        The Stim circuit to analyze.
        
    Returns
    -------
    CircuitAnalysis
        Analysis results including gate counts, depth, parallelism.
    """
    import stim
    
    analysis = CircuitAnalysis()
    
    # Basic properties
    analysis.num_qubits = circuit.num_qubits
    analysis.num_detectors = circuit.num_detectors
    analysis.num_observables = circuit.num_observables
    
    # Get qubit coordinates
    try:
        coords = circuit.get_final_qubit_coordinates()
        if coords:
            analysis.has_qubit_coords = True
            analysis.qubit_coords = coords
    except Exception:
        pass
    
    # Analyze gates - flatten circuit first to handle REPEAT blocks
    flattened = circuit.flattened()
    
    gate_counts: Dict[str, int] = {}
    tick_count = 0
    ops_in_current_tick = 0
    ops_per_tick: List[int] = []
    
    two_qubit_gates = {'CX', 'CY', 'CZ', 'CNOT', 'SWAP', 'ISWAP', 'CXSWAP', 'SWAPCX', 
                       'XCX', 'XCY', 'XCZ', 'YCX', 'YCY', 'YCZ', 'ZCX', 'ZCY', 'ZCZ',
                       'SQRT_XX', 'SQRT_YY', 'SQRT_ZZ', 'MXX', 'MYY', 'MZZ', 'MPP'}
    
    for instruction in flattened:
        name = instruction.name
        
        if name == 'TICK':
            tick_count += 1
            if ops_in_current_tick > 0:
                ops_per_tick.append(ops_in_current_tick)
            ops_in_current_tick = 0
            continue
        
        # Skip annotations
        if name in {'QUBIT_COORDS', 'DETECTOR', 'OBSERVABLE_INCLUDE', 'SHIFT_COORDS'}:
            continue
        
        # Count gate
        gate_counts[name] = gate_counts.get(name, 0) + 1
        
        # Count as operation in this tick
        num_targets = len(instruction.targets_copy())
        if name in two_qubit_gates:
            # Two-qubit gates operate on pairs
            ops_count = num_targets // 2 if num_targets > 0 else 1
            analysis.two_qubit_gates += ops_count
            ops_in_current_tick += ops_count
        elif name.startswith('M') or name in {'H', 'X', 'Y', 'Z', 'S', 'S_DAG', 
                                                'SQRT_X', 'SQRT_Y', 'SQRT_Z',
                                                'R', 'RX', 'RY', 'RZ'}:
            # Single-qubit gates
            ops_count = num_targets
            analysis.single_qubit_gates += ops_count
            ops_in_current_tick += ops_count
    
    # Don't forget last tick
    if ops_in_current_tick > 0:
        ops_per_tick.append(ops_in_current_tick)
    
    analysis.gate_counts = gate_counts
    analysis.num_ticks = tick_count
    analysis.circuit_depth = tick_count  # TICKs define circuit depth
    analysis.ops_per_tick = ops_per_tick
    
    # Compute parallelism metrics
    if ops_per_tick:
        analysis.avg_ops_per_tick = sum(ops_per_tick) / len(ops_per_tick)
        analysis.max_ops_per_tick = max(ops_per_tick)
        
        # Parallelism score: ratio of actual ops per tick vs theoretical max
        # Higher score = more parallel
        total_ops = sum(ops_per_tick)
        if tick_count > 0 and total_ops > 0:
            # Score = avg ops per tick normalized
            # A perfectly serial circuit would have 1 op per tick
            # A perfectly parallel circuit would have all ops in one tick
            analysis.parallelism_score = analysis.avg_ops_per_tick
    
    return analysis


# =============================================================================
# Visualization Helpers
# =============================================================================

def visualize_code_geometry(
    code,
    ax=None,
    show_data: bool = True,
    show_x_stabs: bool = True,
    show_z_stabs: bool = True,
    data_color: str = 'blue',
    x_stab_color: str = 'red',
    z_stab_color: str = 'green',
    title: Optional[str] = None,
):
    """Visualize the geometry of a quantum code.
    
    Parameters
    ----------
    code : CSSCode
        The code to visualize.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, creates new figure.
    show_data : bool
        Whether to show data qubit positions.
    show_x_stabs : bool
        Whether to show X stabilizer positions.
    show_z_stabs : bool
        Whether to show Z stabilizer positions.
    data_color, x_stab_color, z_stab_color : str
        Colors for each qubit type.
    title : str, optional
        Plot title. Defaults to code name.
        
    Returns
    -------
    matplotlib.axes.Axes
        The axes with the plot.
    """
    import matplotlib.pyplot as plt
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
    
    # Get data qubit coordinates
    if hasattr(code, 'qubit_coords'):
        data_coords = code.qubit_coords()
        if data_coords and show_data:
            data_arr = np.array(data_coords)
            ax.scatter(data_arr[:, 0], data_arr[:, 1], 
                      c=data_color, s=100, label='Data', zorder=5)
    
    # Get stabilizer coordinates from metadata
    meta = getattr(code, '_metadata', {})
    
    if show_x_stabs:
        x_stab_coords = meta.get('x_stab_coords', [])
        if x_stab_coords:
            x_arr = np.array(x_stab_coords)
            ax.scatter(x_arr[:, 0], x_arr[:, 1],
                      c=x_stab_color, s=80, marker='s', label='X stab', zorder=4)
    
    if show_z_stabs:
        z_stab_coords = meta.get('z_stab_coords', [])
        if z_stab_coords:
            z_arr = np.array(z_stab_coords)
            ax.scatter(z_arr[:, 0], z_arr[:, 1],
                      c=z_stab_color, s=80, marker='^', label='Z stab', zorder=4)
    
    # Set title
    if title is None:
        title = getattr(code, 'name', type(code).__name__)
        if hasattr(code, 'n'):
            title += f'\nn={code.n}'
    ax.set_title(title)
    
    ax.legend()
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    return ax


def plot_qubit_topology(
    circuit,
    ax=None,
    title: str = "Qubit Topology",
    show_qubit_ids: bool = True,
):
    """Plot qubit topology from Stim circuit QUBIT_COORDS.
    
    Parameters
    ----------
    circuit : stim.Circuit
        The Stim circuit.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on.
    title : str
        Plot title.
    show_qubit_ids : bool
        Whether to annotate qubit indices.
        
    Returns
    -------
    matplotlib.axes.Axes
        The axes with the plot.
    """
    import matplotlib.pyplot as plt
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 10))
    
    coords = circuit.get_final_qubit_coordinates()
    
    if not coords:
        ax.text(0.5, 0.5, "No QUBIT_COORDS found", 
               ha='center', va='center', transform=ax.transAxes)
        ax.set_title(title)
        return ax
    
    # Plot qubits
    for qubit_idx, coord in coords.items():
        x, y = coord[0], coord[1] if len(coord) > 1 else 0
        ax.scatter(x, y, c='blue', s=50, zorder=5)
        if show_qubit_ids:
            ax.annotate(str(qubit_idx), (x, y), fontsize=6, 
                       ha='center', va='bottom', alpha=0.7)
    
    ax.set_title(f"{title} ({len(coords)} qubits)")
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    return ax


# =============================================================================
# Table Formatting
# =============================================================================

def format_status(status: str, value: Optional[float] = None) -> str:
    """Format a status with optional value for table display.
    
    Parameters
    ----------
    status : str
        Status code: 'OK', 'WARN', 'SKIP', 'FAIL'.
    value : float, optional
        Numeric value to display.
        
    Returns
    -------
    str
        Formatted status string.
    """
    indicator = {
        'OK': STATUS_OK,
        'WARN': STATUS_WARN,
        'SKIP': STATUS_SKIP,
        'FAIL': STATUS_FAIL,
    }.get(status, '?')
    
    if value is not None:
        return f"{indicator}{value:.4f}"
    return f"{indicator} {status}"


def print_results_table(
    results: Dict[str, Dict[str, Any]],
    columns: List[str],
    title: str = "Results",
    code_width: int = 35,
    col_width: int = 12,
):
    """Print a formatted results table.
    
    Parameters
    ----------
    results : dict
        Dict mapping code names to column results.
        Each result should have 'status' and optionally 'ler'.
    columns : list
        Column names to display.
    title : str
        Table title.
    code_width : int
        Width of code name column.
    col_width : int
        Width of each result column.
    """
    # Header
    print("=" * (code_width + len(columns) * (col_width + 3)))
    print(title)
    print("=" * (code_width + len(columns) * (col_width + 3)))
    
    header = f"{'Code':<{code_width}}"
    for col in columns:
        header += f" | {col[:col_width]:<{col_width}}"
    print(header)
    print("-" * len(header))
    
    # Rows
    for code_name, code_results in sorted(results.items()):
        row = f"{code_name[:code_width]:<{code_width}}"
        for col in columns:
            res = code_results.get(col, {})
            if isinstance(res, dict):
                status = res.get('status', 'UNKNOWN')
                ler = res.get('ler')
                cell = format_status(status, ler)
            else:
                cell = str(res)[:col_width]
            row += f" | {cell:<{col_width}}"
        print(row)
    
    print("-" * len(header))
    print(f"Total: {len(results)} codes")


# =============================================================================
# Summary Statistics
# =============================================================================

def compute_summary_stats(
    results: Dict[str, Dict[str, Any]],
) -> Dict[str, int]:
    """Compute summary statistics from test results.
    
    Parameters
    ----------
    results : dict
        Dict mapping code names to decoder/test results.
        
    Returns
    -------
    dict
        Summary with 'passed', 'warned', 'skipped', 'failed' counts.
    """
    passed = 0
    warned = 0
    skipped = 0
    failed = 0
    
    for code_results in results.values():
        for key, res in code_results.items():
            if not isinstance(res, dict):
                continue
            status = res.get('status', 'UNKNOWN')
            if status == 'OK':
                passed += 1
            elif status == 'WARN':
                warned += 1
            elif status == 'SKIP':
                skipped += 1
            elif status == 'FAIL':
                failed += 1
    
    return {
        'passed': passed,
        'warned': warned,
        'skipped': skipped,
        'failed': failed,
        'total': passed + warned + skipped + failed,
    }
