# src/qectostim/testing/__init__.py
"""
QECToStim Testing Utilities.

This module provides shared testing infrastructure for QECToStim,
including decoder testing, circuit analysis, and visualization helpers.

Usage
-----
>>> from qectostim.testing import test_decoder_on_code, analyze_stim_circuit
>>> from qectostim.testing import visualize_code_geometry, TestResult

Available Functions
-------------------
- test_decoder_on_code: Test a decoder on any code type
- test_composite_construction: Test composite code building
- test_code_circuit: Verify code generates valid Stim circuit
- analyze_stim_circuit: Extract circuit properties (depth, parallelism, etc.)
- visualize_code_geometry: Plot code qubit/stabilizer layout
- plot_qubit_topology: Plot circuit QUBIT_COORDS
- clear_qectostim_modules: Clear module cache for fresh imports

Status Indicators
-----------------
- STATUS_OK (✓): Test passed
- STATUS_WARN (⚠️): Test passed with warnings
- STATUS_SKIP (-): Test skipped (expected)
- STATUS_FAIL (✗): Test failed
"""

from .testing_utils import (
    # Status indicators
    STATUS_OK,
    STATUS_WARN,
    STATUS_SKIP,
    STATUS_FAIL,
    
    # Module management
    clear_qectostim_modules,
    
    # Test results
    TestResult,
    
    # Testing functions
    test_decoder_on_code,
    test_composite_construction,
    test_code_circuit,
    
    # Circuit analysis
    CircuitAnalysis,
    analyze_stim_circuit,
    
    # Visualization
    visualize_code_geometry,
    plot_qubit_topology,
    
    # Table formatting
    format_status,
    print_results_table,
    compute_summary_stats,
    
    # Decoder loading
    load_all_decoders,
    
    # Code categorization
    categorize_codes,
    discover_and_categorize_codes,
    print_code_summary,
)

__all__ = [
    # Status
    'STATUS_OK',
    'STATUS_WARN',
    'STATUS_SKIP',
    'STATUS_FAIL',
    
    # Module management
    'clear_qectostim_modules',
    
    # Data classes
    'TestResult',
    'CircuitAnalysis',
    
    # Test functions
    'test_decoder_on_code',
    'test_composite_construction',
    'test_code_circuit',
    
    # Analysis
    'analyze_stim_circuit',
    
    # Visualization
    'visualize_code_geometry',
    'plot_qubit_topology',
    
    # Formatting
    'format_status',
    'print_results_table',
    'compute_summary_stats',
    
    # Decoder loading
    'load_all_decoders',
    
    # Code categorization
    'categorize_codes',
    'discover_and_categorize_codes',
    'print_code_summary',
]
