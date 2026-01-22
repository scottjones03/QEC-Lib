"""
Small CSS and stabilizer codes.

This module contains implementations of small, foundational quantum error
correction codes often used for testing and as building blocks.

CSS Codes:
    FourQubit422Code: [[4,2,2]] detection code
    SixQubit622Code: [[6,2,2]] CSS code
    SteaneCode713: [[7,1,3]] Steane code
    ShorCode91: [[9,1,3]] Shor code
    HammingCSSCode: [[7,4,3]] Hamming-based CSS code
    ReedMullerCode151: [[15,1,3]] quantum Reed-Muller code

Non-CSS Codes:
    PerfectCode513: [[5,1,3]] perfect code
    EightThreeTwoCode: [[8,3,2]] code
    SixQubit642Code: [[6,4,2]] code
    BareAncillaCode713: [[7,1,3]] bare ancilla (non-CSS) code
    TenQubitCode: [[10,2,3]] code
    FiveQubitMixedCode: [[5,1,3]] mixed stabilizer code

Repetition Codes:
    RepetitionCode: Parameterized N-qubit repetition code
"""

# CSS codes (now local)
from .four_two_two import FourQubit422Code
from .six_two_two import SixQubit622Code
from .steane_713 import SteaneCode713
from .shor_code import ShorCode91
from .hamming_css import HammingCSSCode, HammingCSS7, HammingCSS15, HammingCSS31
from .reed_muller_code import ReedMullerCode151

# Non-CSS codes (now local)
from .perfect_code import PerfectCode513
from .eight_three_two import EightThreeTwoCode
from .non_css_codes import (
    SixQubit642Code,
    BareAncillaCode713,
    TenQubitCode,
    FiveQubitMixedCode,
    NonCSS642,
    NonCSS713,
    NonCSS1023,
    Mixed512,
)

# Repetition codes (now local)
from .repetition_codes import (
    RepetitionCode,
    create_repetition_code_3,
    create_repetition_code_5,
    create_repetition_code_7,
    create_repetition_code_9,
)

__all__ = [
    # CSS
    "FourQubit422Code",
    "SixQubit622Code",
    "SteaneCode713",
    "ShorCode91",
    "HammingCSSCode",
    "HammingCSS7",
    "HammingCSS15",
    "HammingCSS31",
    "ReedMullerCode151",
    # Non-CSS
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
    # Repetition
    "RepetitionCode",
    "create_repetition_code_3",
    "create_repetition_code_5",
    "create_repetition_code_7",
    "create_repetition_code_9",
]
