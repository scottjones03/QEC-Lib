# src/tests/__init__.py
"""
QECToStim Testing Utilities.

DEPRECATED: Use `from qectostim.testing import ...` instead.

This module is deprecated because it conflicts with the `tests` package
installed in the virtual environment. Please update your imports to:

    from qectostim.testing import (
        test_decoder_on_code,
        analyze_stim_circuit,
        visualize_code_geometry,
        ...
    )
"""

import warnings as _warnings
_warnings.warn(
    "Importing from 'tests' is deprecated. Use 'from qectostim.testing import ...' instead.",
    DeprecationWarning,
    stacklevel=2,
)

# Re-export from qectostim.testing for backward compatibility
from qectostim.testing import (
    STATUS_OK, STATUS_WARN, STATUS_SKIP, STATUS_FAIL,
    TestResult, CircuitAnalysis,
    clear_qectostim_modules, test_decoder_on_code,
    test_composite_construction, test_code_circuit,
    analyze_stim_circuit, visualize_code_geometry,
    plot_qubit_topology, format_status, print_results_table,
    compute_summary_stats,
)

__all__ = [
    'STATUS_OK', 'STATUS_WARN', 'STATUS_SKIP', 'STATUS_FAIL',
    'TestResult', 'CircuitAnalysis',
    'clear_qectostim_modules', 'test_decoder_on_code',
    'test_composite_construction', 'test_code_circuit',
    'analyze_stim_circuit', 'visualize_code_geometry',
    'plot_qubit_topology', 'format_status', 'print_results_table',
    'compute_summary_stats',
]
